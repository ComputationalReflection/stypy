
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy import array, kron, matrix, diag
4: from numpy.testing import assert_, assert_equal
5: 
6: from scipy.sparse import spfuncs
7: from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix
8: from scipy.sparse._sparsetools import (csr_scale_rows, csr_scale_columns,
9:                                        bsr_scale_rows, bsr_scale_columns)
10: 
11: 
12: class TestSparseFunctions(object):
13:     def test_scale_rows_and_cols(self):
14:         D = matrix([[1,0,0,2,3],
15:                     [0,4,0,5,0],
16:                     [0,0,6,7,0]])
17: 
18:         #TODO expose through function
19:         S = csr_matrix(D)
20:         v = array([1,2,3])
21:         csr_scale_rows(3,5,S.indptr,S.indices,S.data,v)
22:         assert_equal(S.todense(), diag(v)*D)
23: 
24:         S = csr_matrix(D)
25:         v = array([1,2,3,4,5])
26:         csr_scale_columns(3,5,S.indptr,S.indices,S.data,v)
27:         assert_equal(S.todense(), D*diag(v))
28: 
29:         # blocks
30:         E = kron(D,[[1,2],[3,4]])
31:         S = bsr_matrix(E,blocksize=(2,2))
32:         v = array([1,2,3,4,5,6])
33:         bsr_scale_rows(3,5,2,2,S.indptr,S.indices,S.data,v)
34:         assert_equal(S.todense(), diag(v)*E)
35: 
36:         S = bsr_matrix(E,blocksize=(2,2))
37:         v = array([1,2,3,4,5,6,7,8,9,10])
38:         bsr_scale_columns(3,5,2,2,S.indptr,S.indices,S.data,v)
39:         assert_equal(S.todense(), E*diag(v))
40: 
41:         E = kron(D,[[1,2,3],[4,5,6]])
42:         S = bsr_matrix(E,blocksize=(2,3))
43:         v = array([1,2,3,4,5,6])
44:         bsr_scale_rows(3,5,2,3,S.indptr,S.indices,S.data,v)
45:         assert_equal(S.todense(), diag(v)*E)
46: 
47:         S = bsr_matrix(E,blocksize=(2,3))
48:         v = array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
49:         bsr_scale_columns(3,5,2,3,S.indptr,S.indices,S.data,v)
50:         assert_equal(S.todense(), E*diag(v))
51: 
52:     def test_estimate_blocksize(self):
53:         mats = []
54:         mats.append([[0,1],[1,0]])
55:         mats.append([[1,1,0],[0,0,1],[1,0,1]])
56:         mats.append([[0],[0],[1]])
57:         mats = [array(x) for x in mats]
58: 
59:         blks = []
60:         blks.append([[1]])
61:         blks.append([[1,1],[1,1]])
62:         blks.append([[1,1],[0,1]])
63:         blks.append([[1,1,0],[1,0,1],[1,1,1]])
64:         blks = [array(x) for x in blks]
65: 
66:         for A in mats:
67:             for B in blks:
68:                 X = kron(A,B)
69:                 r,c = spfuncs.estimate_blocksize(X)
70:                 assert_(r >= B.shape[0])
71:                 assert_(c >= B.shape[1])
72: 
73:     def test_count_blocks(self):
74:         def gold(A,bs):
75:             R,C = bs
76:             I,J = A.nonzero()
77:             return len(set(zip(I//R,J//C)))
78: 
79:         mats = []
80:         mats.append([[0]])
81:         mats.append([[1]])
82:         mats.append([[1,0]])
83:         mats.append([[1,1]])
84:         mats.append([[0,1],[1,0]])
85:         mats.append([[1,1,0],[0,0,1],[1,0,1]])
86:         mats.append([[0],[0],[1]])
87: 
88:         for A in mats:
89:             for B in mats:
90:                 X = kron(A,B)
91:                 Y = csr_matrix(X)
92:                 for R in range(1,6):
93:                     for C in range(1,6):
94:                         assert_equal(spfuncs.count_blocks(Y, (R, C)), gold(X, (R, C)))
95: 
96:         X = kron([[1,1,0],[0,0,1],[1,0,1]],[[1,1]])
97:         Y = csc_matrix(X)
98:         assert_equal(spfuncs.count_blocks(X, (1, 2)), gold(X, (1, 2)))
99:         assert_equal(spfuncs.count_blocks(Y, (1, 2)), gold(X, (1, 2)))
100: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy import array, kron, matrix, diag' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_461526 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_461526) is not StypyTypeError):

    if (import_461526 != 'pyd_module'):
        __import__(import_461526)
        sys_modules_461527 = sys.modules[import_461526]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', sys_modules_461527.module_type_store, module_type_store, ['array', 'kron', 'matrix', 'diag'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_461527, sys_modules_461527.module_type_store, module_type_store)
    else:
        from numpy import array, kron, matrix, diag

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', None, module_type_store, ['array', 'kron', 'matrix', 'diag'], [array, kron, matrix, diag])

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_461526)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_, assert_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_461528 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_461528) is not StypyTypeError):

    if (import_461528 != 'pyd_module'):
        __import__(import_461528)
        sys_modules_461529 = sys.modules[import_461528]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_461529.module_type_store, module_type_store, ['assert_', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_461529, sys_modules_461529.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal'], [assert_, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_461528)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse import spfuncs' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_461530 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse')

if (type(import_461530) is not StypyTypeError):

    if (import_461530 != 'pyd_module'):
        __import__(import_461530)
        sys_modules_461531 = sys.modules[import_461530]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', sys_modules_461531.module_type_store, module_type_store, ['spfuncs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_461531, sys_modules_461531.module_type_store, module_type_store)
    else:
        from scipy.sparse import spfuncs

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', None, module_type_store, ['spfuncs'], [spfuncs])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', import_461530)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_461532 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse')

if (type(import_461532) is not StypyTypeError):

    if (import_461532 != 'pyd_module'):
        __import__(import_461532)
        sys_modules_461533 = sys.modules[import_461532]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', sys_modules_461533.module_type_store, module_type_store, ['csr_matrix', 'csc_matrix', 'bsr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_461533, sys_modules_461533.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix', 'csc_matrix', 'bsr_matrix'], [csr_matrix, csc_matrix, bsr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', import_461532)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.sparse._sparsetools import csr_scale_rows, csr_scale_columns, bsr_scale_rows, bsr_scale_columns' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_461534 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse._sparsetools')

if (type(import_461534) is not StypyTypeError):

    if (import_461534 != 'pyd_module'):
        __import__(import_461534)
        sys_modules_461535 = sys.modules[import_461534]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse._sparsetools', sys_modules_461535.module_type_store, module_type_store, ['csr_scale_rows', 'csr_scale_columns', 'bsr_scale_rows', 'bsr_scale_columns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_461535, sys_modules_461535.module_type_store, module_type_store)
    else:
        from scipy.sparse._sparsetools import csr_scale_rows, csr_scale_columns, bsr_scale_rows, bsr_scale_columns

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse._sparsetools', None, module_type_store, ['csr_scale_rows', 'csr_scale_columns', 'bsr_scale_rows', 'bsr_scale_columns'], [csr_scale_rows, csr_scale_columns, bsr_scale_rows, bsr_scale_columns])

else:
    # Assigning a type to the variable 'scipy.sparse._sparsetools' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse._sparsetools', import_461534)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

# Declaration of the 'TestSparseFunctions' class

class TestSparseFunctions(object, ):

    @norecursion
    def test_scale_rows_and_cols(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scale_rows_and_cols'
        module_type_store = module_type_store.open_function_context('test_scale_rows_and_cols', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_localization', localization)
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_function_name', 'TestSparseFunctions.test_scale_rows_and_cols')
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseFunctions.test_scale_rows_and_cols.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseFunctions.test_scale_rows_and_cols', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scale_rows_and_cols', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scale_rows_and_cols(...)' code ##################

        
        # Assigning a Call to a Name (line 14):
        
        # Assigning a Call to a Name (line 14):
        
        # Call to matrix(...): (line 14)
        # Processing the call arguments (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_461537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_461538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        int_461539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 20), list_461538, int_461539)
        # Adding element type (line 14)
        int_461540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 20), list_461538, int_461540)
        # Adding element type (line 14)
        int_461541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 20), list_461538, int_461541)
        # Adding element type (line 14)
        int_461542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 20), list_461538, int_461542)
        # Adding element type (line 14)
        int_461543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 20), list_461538, int_461543)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 19), list_461537, list_461538)
        # Adding element type (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_461544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        # Adding element type (line 15)
        int_461545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_461544, int_461545)
        # Adding element type (line 15)
        int_461546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_461544, int_461546)
        # Adding element type (line 15)
        int_461547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_461544, int_461547)
        # Adding element type (line 15)
        int_461548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_461544, int_461548)
        # Adding element type (line 15)
        int_461549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_461544, int_461549)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 19), list_461537, list_461544)
        # Adding element type (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_461550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        # Adding element type (line 16)
        int_461551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_461550, int_461551)
        # Adding element type (line 16)
        int_461552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_461550, int_461552)
        # Adding element type (line 16)
        int_461553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_461550, int_461553)
        # Adding element type (line 16)
        int_461554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_461550, int_461554)
        # Adding element type (line 16)
        int_461555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_461550, int_461555)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 19), list_461537, list_461550)
        
        # Processing the call keyword arguments (line 14)
        kwargs_461556 = {}
        # Getting the type of 'matrix' (line 14)
        matrix_461536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'matrix', False)
        # Calling matrix(args, kwargs) (line 14)
        matrix_call_result_461557 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), matrix_461536, *[list_461537], **kwargs_461556)
        
        # Assigning a type to the variable 'D' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'D', matrix_call_result_461557)
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to csr_matrix(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'D' (line 19)
        D_461559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'D', False)
        # Processing the call keyword arguments (line 19)
        kwargs_461560 = {}
        # Getting the type of 'csr_matrix' (line 19)
        csr_matrix_461558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 19)
        csr_matrix_call_result_461561 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), csr_matrix_461558, *[D_461559], **kwargs_461560)
        
        # Assigning a type to the variable 'S' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'S', csr_matrix_call_result_461561)
        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to array(...): (line 20)
        # Processing the call arguments (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_461563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        int_461564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), list_461563, int_461564)
        # Adding element type (line 20)
        int_461565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), list_461563, int_461565)
        # Adding element type (line 20)
        int_461566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), list_461563, int_461566)
        
        # Processing the call keyword arguments (line 20)
        kwargs_461567 = {}
        # Getting the type of 'array' (line 20)
        array_461562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'array', False)
        # Calling array(args, kwargs) (line 20)
        array_call_result_461568 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), array_461562, *[list_461563], **kwargs_461567)
        
        # Assigning a type to the variable 'v' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'v', array_call_result_461568)
        
        # Call to csr_scale_rows(...): (line 21)
        # Processing the call arguments (line 21)
        int_461570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'int')
        int_461571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'int')
        # Getting the type of 'S' (line 21)
        S_461572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'S', False)
        # Obtaining the member 'indptr' of a type (line 21)
        indptr_461573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 27), S_461572, 'indptr')
        # Getting the type of 'S' (line 21)
        S_461574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'S', False)
        # Obtaining the member 'indices' of a type (line 21)
        indices_461575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 36), S_461574, 'indices')
        # Getting the type of 'S' (line 21)
        S_461576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 46), 'S', False)
        # Obtaining the member 'data' of a type (line 21)
        data_461577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 46), S_461576, 'data')
        # Getting the type of 'v' (line 21)
        v_461578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 53), 'v', False)
        # Processing the call keyword arguments (line 21)
        kwargs_461579 = {}
        # Getting the type of 'csr_scale_rows' (line 21)
        csr_scale_rows_461569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'csr_scale_rows', False)
        # Calling csr_scale_rows(args, kwargs) (line 21)
        csr_scale_rows_call_result_461580 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), csr_scale_rows_461569, *[int_461570, int_461571, indptr_461573, indices_461575, data_461577, v_461578], **kwargs_461579)
        
        
        # Call to assert_equal(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Call to todense(...): (line 22)
        # Processing the call keyword arguments (line 22)
        kwargs_461584 = {}
        # Getting the type of 'S' (line 22)
        S_461582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'S', False)
        # Obtaining the member 'todense' of a type (line 22)
        todense_461583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 21), S_461582, 'todense')
        # Calling todense(args, kwargs) (line 22)
        todense_call_result_461585 = invoke(stypy.reporting.localization.Localization(__file__, 22, 21), todense_461583, *[], **kwargs_461584)
        
        
        # Call to diag(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'v' (line 22)
        v_461587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'v', False)
        # Processing the call keyword arguments (line 22)
        kwargs_461588 = {}
        # Getting the type of 'diag' (line 22)
        diag_461586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'diag', False)
        # Calling diag(args, kwargs) (line 22)
        diag_call_result_461589 = invoke(stypy.reporting.localization.Localization(__file__, 22, 34), diag_461586, *[v_461587], **kwargs_461588)
        
        # Getting the type of 'D' (line 22)
        D_461590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 42), 'D', False)
        # Applying the binary operator '*' (line 22)
        result_mul_461591 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 34), '*', diag_call_result_461589, D_461590)
        
        # Processing the call keyword arguments (line 22)
        kwargs_461592 = {}
        # Getting the type of 'assert_equal' (line 22)
        assert_equal_461581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 22)
        assert_equal_call_result_461593 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), assert_equal_461581, *[todense_call_result_461585, result_mul_461591], **kwargs_461592)
        
        
        # Assigning a Call to a Name (line 24):
        
        # Assigning a Call to a Name (line 24):
        
        # Call to csr_matrix(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'D' (line 24)
        D_461595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'D', False)
        # Processing the call keyword arguments (line 24)
        kwargs_461596 = {}
        # Getting the type of 'csr_matrix' (line 24)
        csr_matrix_461594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 24)
        csr_matrix_call_result_461597 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), csr_matrix_461594, *[D_461595], **kwargs_461596)
        
        # Assigning a type to the variable 'S' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'S', csr_matrix_call_result_461597)
        
        # Assigning a Call to a Name (line 25):
        
        # Assigning a Call to a Name (line 25):
        
        # Call to array(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_461599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_461600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 18), list_461599, int_461600)
        # Adding element type (line 25)
        int_461601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 18), list_461599, int_461601)
        # Adding element type (line 25)
        int_461602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 18), list_461599, int_461602)
        # Adding element type (line 25)
        int_461603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 18), list_461599, int_461603)
        # Adding element type (line 25)
        int_461604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 18), list_461599, int_461604)
        
        # Processing the call keyword arguments (line 25)
        kwargs_461605 = {}
        # Getting the type of 'array' (line 25)
        array_461598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'array', False)
        # Calling array(args, kwargs) (line 25)
        array_call_result_461606 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), array_461598, *[list_461599], **kwargs_461605)
        
        # Assigning a type to the variable 'v' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'v', array_call_result_461606)
        
        # Call to csr_scale_columns(...): (line 26)
        # Processing the call arguments (line 26)
        int_461608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'int')
        int_461609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
        # Getting the type of 'S' (line 26)
        S_461610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'S', False)
        # Obtaining the member 'indptr' of a type (line 26)
        indptr_461611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 30), S_461610, 'indptr')
        # Getting the type of 'S' (line 26)
        S_461612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 39), 'S', False)
        # Obtaining the member 'indices' of a type (line 26)
        indices_461613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 39), S_461612, 'indices')
        # Getting the type of 'S' (line 26)
        S_461614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 49), 'S', False)
        # Obtaining the member 'data' of a type (line 26)
        data_461615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 49), S_461614, 'data')
        # Getting the type of 'v' (line 26)
        v_461616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 56), 'v', False)
        # Processing the call keyword arguments (line 26)
        kwargs_461617 = {}
        # Getting the type of 'csr_scale_columns' (line 26)
        csr_scale_columns_461607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'csr_scale_columns', False)
        # Calling csr_scale_columns(args, kwargs) (line 26)
        csr_scale_columns_call_result_461618 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), csr_scale_columns_461607, *[int_461608, int_461609, indptr_461611, indices_461613, data_461615, v_461616], **kwargs_461617)
        
        
        # Call to assert_equal(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to todense(...): (line 27)
        # Processing the call keyword arguments (line 27)
        kwargs_461622 = {}
        # Getting the type of 'S' (line 27)
        S_461620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'S', False)
        # Obtaining the member 'todense' of a type (line 27)
        todense_461621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), S_461620, 'todense')
        # Calling todense(args, kwargs) (line 27)
        todense_call_result_461623 = invoke(stypy.reporting.localization.Localization(__file__, 27, 21), todense_461621, *[], **kwargs_461622)
        
        # Getting the type of 'D' (line 27)
        D_461624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'D', False)
        
        # Call to diag(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'v' (line 27)
        v_461626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 41), 'v', False)
        # Processing the call keyword arguments (line 27)
        kwargs_461627 = {}
        # Getting the type of 'diag' (line 27)
        diag_461625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 36), 'diag', False)
        # Calling diag(args, kwargs) (line 27)
        diag_call_result_461628 = invoke(stypy.reporting.localization.Localization(__file__, 27, 36), diag_461625, *[v_461626], **kwargs_461627)
        
        # Applying the binary operator '*' (line 27)
        result_mul_461629 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 34), '*', D_461624, diag_call_result_461628)
        
        # Processing the call keyword arguments (line 27)
        kwargs_461630 = {}
        # Getting the type of 'assert_equal' (line 27)
        assert_equal_461619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 27)
        assert_equal_call_result_461631 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_equal_461619, *[todense_call_result_461623, result_mul_461629], **kwargs_461630)
        
        
        # Assigning a Call to a Name (line 30):
        
        # Assigning a Call to a Name (line 30):
        
        # Call to kron(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'D' (line 30)
        D_461633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'D', False)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_461634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_461635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        int_461636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 20), list_461635, int_461636)
        # Adding element type (line 30)
        int_461637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 20), list_461635, int_461637)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 19), list_461634, list_461635)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_461638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        int_461639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_461638, int_461639)
        # Adding element type (line 30)
        int_461640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_461638, int_461640)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 19), list_461634, list_461638)
        
        # Processing the call keyword arguments (line 30)
        kwargs_461641 = {}
        # Getting the type of 'kron' (line 30)
        kron_461632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'kron', False)
        # Calling kron(args, kwargs) (line 30)
        kron_call_result_461642 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), kron_461632, *[D_461633, list_461634], **kwargs_461641)
        
        # Assigning a type to the variable 'E' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'E', kron_call_result_461642)
        
        # Assigning a Call to a Name (line 31):
        
        # Assigning a Call to a Name (line 31):
        
        # Call to bsr_matrix(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'E' (line 31)
        E_461644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'E', False)
        # Processing the call keyword arguments (line 31)
        
        # Obtaining an instance of the builtin type 'tuple' (line 31)
        tuple_461645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 31)
        # Adding element type (line 31)
        int_461646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 36), tuple_461645, int_461646)
        # Adding element type (line 31)
        int_461647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 36), tuple_461645, int_461647)
        
        keyword_461648 = tuple_461645
        kwargs_461649 = {'blocksize': keyword_461648}
        # Getting the type of 'bsr_matrix' (line 31)
        bsr_matrix_461643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 31)
        bsr_matrix_call_result_461650 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), bsr_matrix_461643, *[E_461644], **kwargs_461649)
        
        # Assigning a type to the variable 'S' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'S', bsr_matrix_call_result_461650)
        
        # Assigning a Call to a Name (line 32):
        
        # Assigning a Call to a Name (line 32):
        
        # Call to array(...): (line 32)
        # Processing the call arguments (line 32)
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_461652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        int_461653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_461652, int_461653)
        # Adding element type (line 32)
        int_461654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_461652, int_461654)
        # Adding element type (line 32)
        int_461655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_461652, int_461655)
        # Adding element type (line 32)
        int_461656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_461652, int_461656)
        # Adding element type (line 32)
        int_461657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_461652, int_461657)
        # Adding element type (line 32)
        int_461658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_461652, int_461658)
        
        # Processing the call keyword arguments (line 32)
        kwargs_461659 = {}
        # Getting the type of 'array' (line 32)
        array_461651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'array', False)
        # Calling array(args, kwargs) (line 32)
        array_call_result_461660 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), array_461651, *[list_461652], **kwargs_461659)
        
        # Assigning a type to the variable 'v' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'v', array_call_result_461660)
        
        # Call to bsr_scale_rows(...): (line 33)
        # Processing the call arguments (line 33)
        int_461662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'int')
        int_461663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'int')
        int_461664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 27), 'int')
        int_461665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 29), 'int')
        # Getting the type of 'S' (line 33)
        S_461666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'S', False)
        # Obtaining the member 'indptr' of a type (line 33)
        indptr_461667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 31), S_461666, 'indptr')
        # Getting the type of 'S' (line 33)
        S_461668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'S', False)
        # Obtaining the member 'indices' of a type (line 33)
        indices_461669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 40), S_461668, 'indices')
        # Getting the type of 'S' (line 33)
        S_461670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 50), 'S', False)
        # Obtaining the member 'data' of a type (line 33)
        data_461671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 50), S_461670, 'data')
        # Getting the type of 'v' (line 33)
        v_461672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 57), 'v', False)
        # Processing the call keyword arguments (line 33)
        kwargs_461673 = {}
        # Getting the type of 'bsr_scale_rows' (line 33)
        bsr_scale_rows_461661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'bsr_scale_rows', False)
        # Calling bsr_scale_rows(args, kwargs) (line 33)
        bsr_scale_rows_call_result_461674 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), bsr_scale_rows_461661, *[int_461662, int_461663, int_461664, int_461665, indptr_461667, indices_461669, data_461671, v_461672], **kwargs_461673)
        
        
        # Call to assert_equal(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to todense(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_461678 = {}
        # Getting the type of 'S' (line 34)
        S_461676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'S', False)
        # Obtaining the member 'todense' of a type (line 34)
        todense_461677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), S_461676, 'todense')
        # Calling todense(args, kwargs) (line 34)
        todense_call_result_461679 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), todense_461677, *[], **kwargs_461678)
        
        
        # Call to diag(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'v' (line 34)
        v_461681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 39), 'v', False)
        # Processing the call keyword arguments (line 34)
        kwargs_461682 = {}
        # Getting the type of 'diag' (line 34)
        diag_461680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'diag', False)
        # Calling diag(args, kwargs) (line 34)
        diag_call_result_461683 = invoke(stypy.reporting.localization.Localization(__file__, 34, 34), diag_461680, *[v_461681], **kwargs_461682)
        
        # Getting the type of 'E' (line 34)
        E_461684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'E', False)
        # Applying the binary operator '*' (line 34)
        result_mul_461685 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 34), '*', diag_call_result_461683, E_461684)
        
        # Processing the call keyword arguments (line 34)
        kwargs_461686 = {}
        # Getting the type of 'assert_equal' (line 34)
        assert_equal_461675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 34)
        assert_equal_call_result_461687 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert_equal_461675, *[todense_call_result_461679, result_mul_461685], **kwargs_461686)
        
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to bsr_matrix(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'E' (line 36)
        E_461689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'E', False)
        # Processing the call keyword arguments (line 36)
        
        # Obtaining an instance of the builtin type 'tuple' (line 36)
        tuple_461690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 36)
        # Adding element type (line 36)
        int_461691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 36), tuple_461690, int_461691)
        # Adding element type (line 36)
        int_461692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 36), tuple_461690, int_461692)
        
        keyword_461693 = tuple_461690
        kwargs_461694 = {'blocksize': keyword_461693}
        # Getting the type of 'bsr_matrix' (line 36)
        bsr_matrix_461688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 36)
        bsr_matrix_call_result_461695 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), bsr_matrix_461688, *[E_461689], **kwargs_461694)
        
        # Assigning a type to the variable 'S' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'S', bsr_matrix_call_result_461695)
        
        # Assigning a Call to a Name (line 37):
        
        # Assigning a Call to a Name (line 37):
        
        # Call to array(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_461697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        # Adding element type (line 37)
        int_461698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461698)
        # Adding element type (line 37)
        int_461699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461699)
        # Adding element type (line 37)
        int_461700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461700)
        # Adding element type (line 37)
        int_461701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461701)
        # Adding element type (line 37)
        int_461702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461702)
        # Adding element type (line 37)
        int_461703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461703)
        # Adding element type (line 37)
        int_461704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461704)
        # Adding element type (line 37)
        int_461705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461705)
        # Adding element type (line 37)
        int_461706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461706)
        # Adding element type (line 37)
        int_461707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_461697, int_461707)
        
        # Processing the call keyword arguments (line 37)
        kwargs_461708 = {}
        # Getting the type of 'array' (line 37)
        array_461696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'array', False)
        # Calling array(args, kwargs) (line 37)
        array_call_result_461709 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), array_461696, *[list_461697], **kwargs_461708)
        
        # Assigning a type to the variable 'v' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'v', array_call_result_461709)
        
        # Call to bsr_scale_columns(...): (line 38)
        # Processing the call arguments (line 38)
        int_461711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'int')
        int_461712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'int')
        int_461713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
        int_461714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 32), 'int')
        # Getting the type of 'S' (line 38)
        S_461715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'S', False)
        # Obtaining the member 'indptr' of a type (line 38)
        indptr_461716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 34), S_461715, 'indptr')
        # Getting the type of 'S' (line 38)
        S_461717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 43), 'S', False)
        # Obtaining the member 'indices' of a type (line 38)
        indices_461718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 43), S_461717, 'indices')
        # Getting the type of 'S' (line 38)
        S_461719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 53), 'S', False)
        # Obtaining the member 'data' of a type (line 38)
        data_461720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 53), S_461719, 'data')
        # Getting the type of 'v' (line 38)
        v_461721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 60), 'v', False)
        # Processing the call keyword arguments (line 38)
        kwargs_461722 = {}
        # Getting the type of 'bsr_scale_columns' (line 38)
        bsr_scale_columns_461710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'bsr_scale_columns', False)
        # Calling bsr_scale_columns(args, kwargs) (line 38)
        bsr_scale_columns_call_result_461723 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), bsr_scale_columns_461710, *[int_461711, int_461712, int_461713, int_461714, indptr_461716, indices_461718, data_461720, v_461721], **kwargs_461722)
        
        
        # Call to assert_equal(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to todense(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_461727 = {}
        # Getting the type of 'S' (line 39)
        S_461725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'S', False)
        # Obtaining the member 'todense' of a type (line 39)
        todense_461726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 21), S_461725, 'todense')
        # Calling todense(args, kwargs) (line 39)
        todense_call_result_461728 = invoke(stypy.reporting.localization.Localization(__file__, 39, 21), todense_461726, *[], **kwargs_461727)
        
        # Getting the type of 'E' (line 39)
        E_461729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'E', False)
        
        # Call to diag(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'v' (line 39)
        v_461731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'v', False)
        # Processing the call keyword arguments (line 39)
        kwargs_461732 = {}
        # Getting the type of 'diag' (line 39)
        diag_461730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 36), 'diag', False)
        # Calling diag(args, kwargs) (line 39)
        diag_call_result_461733 = invoke(stypy.reporting.localization.Localization(__file__, 39, 36), diag_461730, *[v_461731], **kwargs_461732)
        
        # Applying the binary operator '*' (line 39)
        result_mul_461734 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 34), '*', E_461729, diag_call_result_461733)
        
        # Processing the call keyword arguments (line 39)
        kwargs_461735 = {}
        # Getting the type of 'assert_equal' (line 39)
        assert_equal_461724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 39)
        assert_equal_call_result_461736 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert_equal_461724, *[todense_call_result_461728, result_mul_461734], **kwargs_461735)
        
        
        # Assigning a Call to a Name (line 41):
        
        # Assigning a Call to a Name (line 41):
        
        # Call to kron(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'D' (line 41)
        D_461738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'D', False)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_461739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_461740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        int_461741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), list_461740, int_461741)
        # Adding element type (line 41)
        int_461742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), list_461740, int_461742)
        # Adding element type (line 41)
        int_461743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), list_461740, int_461743)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 19), list_461739, list_461740)
        # Adding element type (line 41)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_461744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        int_461745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), list_461744, int_461745)
        # Adding element type (line 41)
        int_461746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), list_461744, int_461746)
        # Adding element type (line 41)
        int_461747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), list_461744, int_461747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 19), list_461739, list_461744)
        
        # Processing the call keyword arguments (line 41)
        kwargs_461748 = {}
        # Getting the type of 'kron' (line 41)
        kron_461737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'kron', False)
        # Calling kron(args, kwargs) (line 41)
        kron_call_result_461749 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), kron_461737, *[D_461738, list_461739], **kwargs_461748)
        
        # Assigning a type to the variable 'E' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'E', kron_call_result_461749)
        
        # Assigning a Call to a Name (line 42):
        
        # Assigning a Call to a Name (line 42):
        
        # Call to bsr_matrix(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'E' (line 42)
        E_461751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'E', False)
        # Processing the call keyword arguments (line 42)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_461752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        int_461753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 36), tuple_461752, int_461753)
        # Adding element type (line 42)
        int_461754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 36), tuple_461752, int_461754)
        
        keyword_461755 = tuple_461752
        kwargs_461756 = {'blocksize': keyword_461755}
        # Getting the type of 'bsr_matrix' (line 42)
        bsr_matrix_461750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 42)
        bsr_matrix_call_result_461757 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), bsr_matrix_461750, *[E_461751], **kwargs_461756)
        
        # Assigning a type to the variable 'S' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'S', bsr_matrix_call_result_461757)
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to array(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_461759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        int_461760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), list_461759, int_461760)
        # Adding element type (line 43)
        int_461761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), list_461759, int_461761)
        # Adding element type (line 43)
        int_461762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), list_461759, int_461762)
        # Adding element type (line 43)
        int_461763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), list_461759, int_461763)
        # Adding element type (line 43)
        int_461764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), list_461759, int_461764)
        # Adding element type (line 43)
        int_461765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), list_461759, int_461765)
        
        # Processing the call keyword arguments (line 43)
        kwargs_461766 = {}
        # Getting the type of 'array' (line 43)
        array_461758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'array', False)
        # Calling array(args, kwargs) (line 43)
        array_call_result_461767 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), array_461758, *[list_461759], **kwargs_461766)
        
        # Assigning a type to the variable 'v' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'v', array_call_result_461767)
        
        # Call to bsr_scale_rows(...): (line 44)
        # Processing the call arguments (line 44)
        int_461769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
        int_461770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'int')
        int_461771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'int')
        int_461772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 29), 'int')
        # Getting the type of 'S' (line 44)
        S_461773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'S', False)
        # Obtaining the member 'indptr' of a type (line 44)
        indptr_461774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 31), S_461773, 'indptr')
        # Getting the type of 'S' (line 44)
        S_461775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 40), 'S', False)
        # Obtaining the member 'indices' of a type (line 44)
        indices_461776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 40), S_461775, 'indices')
        # Getting the type of 'S' (line 44)
        S_461777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 50), 'S', False)
        # Obtaining the member 'data' of a type (line 44)
        data_461778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 50), S_461777, 'data')
        # Getting the type of 'v' (line 44)
        v_461779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 57), 'v', False)
        # Processing the call keyword arguments (line 44)
        kwargs_461780 = {}
        # Getting the type of 'bsr_scale_rows' (line 44)
        bsr_scale_rows_461768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'bsr_scale_rows', False)
        # Calling bsr_scale_rows(args, kwargs) (line 44)
        bsr_scale_rows_call_result_461781 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), bsr_scale_rows_461768, *[int_461769, int_461770, int_461771, int_461772, indptr_461774, indices_461776, data_461778, v_461779], **kwargs_461780)
        
        
        # Call to assert_equal(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to todense(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_461785 = {}
        # Getting the type of 'S' (line 45)
        S_461783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'S', False)
        # Obtaining the member 'todense' of a type (line 45)
        todense_461784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 21), S_461783, 'todense')
        # Calling todense(args, kwargs) (line 45)
        todense_call_result_461786 = invoke(stypy.reporting.localization.Localization(__file__, 45, 21), todense_461784, *[], **kwargs_461785)
        
        
        # Call to diag(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'v' (line 45)
        v_461788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'v', False)
        # Processing the call keyword arguments (line 45)
        kwargs_461789 = {}
        # Getting the type of 'diag' (line 45)
        diag_461787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'diag', False)
        # Calling diag(args, kwargs) (line 45)
        diag_call_result_461790 = invoke(stypy.reporting.localization.Localization(__file__, 45, 34), diag_461787, *[v_461788], **kwargs_461789)
        
        # Getting the type of 'E' (line 45)
        E_461791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 42), 'E', False)
        # Applying the binary operator '*' (line 45)
        result_mul_461792 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 34), '*', diag_call_result_461790, E_461791)
        
        # Processing the call keyword arguments (line 45)
        kwargs_461793 = {}
        # Getting the type of 'assert_equal' (line 45)
        assert_equal_461782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 45)
        assert_equal_call_result_461794 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assert_equal_461782, *[todense_call_result_461786, result_mul_461792], **kwargs_461793)
        
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to bsr_matrix(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'E' (line 47)
        E_461796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'E', False)
        # Processing the call keyword arguments (line 47)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_461797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        int_461798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 36), tuple_461797, int_461798)
        # Adding element type (line 47)
        int_461799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 36), tuple_461797, int_461799)
        
        keyword_461800 = tuple_461797
        kwargs_461801 = {'blocksize': keyword_461800}
        # Getting the type of 'bsr_matrix' (line 47)
        bsr_matrix_461795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 47)
        bsr_matrix_call_result_461802 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), bsr_matrix_461795, *[E_461796], **kwargs_461801)
        
        # Assigning a type to the variable 'S' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'S', bsr_matrix_call_result_461802)
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to array(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_461804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        int_461805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461805)
        # Adding element type (line 48)
        int_461806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461806)
        # Adding element type (line 48)
        int_461807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461807)
        # Adding element type (line 48)
        int_461808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461808)
        # Adding element type (line 48)
        int_461809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461809)
        # Adding element type (line 48)
        int_461810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461810)
        # Adding element type (line 48)
        int_461811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461811)
        # Adding element type (line 48)
        int_461812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461812)
        # Adding element type (line 48)
        int_461813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461813)
        # Adding element type (line 48)
        int_461814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461814)
        # Adding element type (line 48)
        int_461815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461815)
        # Adding element type (line 48)
        int_461816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461816)
        # Adding element type (line 48)
        int_461817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461817)
        # Adding element type (line 48)
        int_461818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461818)
        # Adding element type (line 48)
        int_461819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_461804, int_461819)
        
        # Processing the call keyword arguments (line 48)
        kwargs_461820 = {}
        # Getting the type of 'array' (line 48)
        array_461803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'array', False)
        # Calling array(args, kwargs) (line 48)
        array_call_result_461821 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), array_461803, *[list_461804], **kwargs_461820)
        
        # Assigning a type to the variable 'v' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'v', array_call_result_461821)
        
        # Call to bsr_scale_columns(...): (line 49)
        # Processing the call arguments (line 49)
        int_461823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 26), 'int')
        int_461824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'int')
        int_461825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'int')
        int_461826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 32), 'int')
        # Getting the type of 'S' (line 49)
        S_461827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'S', False)
        # Obtaining the member 'indptr' of a type (line 49)
        indptr_461828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 34), S_461827, 'indptr')
        # Getting the type of 'S' (line 49)
        S_461829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'S', False)
        # Obtaining the member 'indices' of a type (line 49)
        indices_461830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 43), S_461829, 'indices')
        # Getting the type of 'S' (line 49)
        S_461831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 53), 'S', False)
        # Obtaining the member 'data' of a type (line 49)
        data_461832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 53), S_461831, 'data')
        # Getting the type of 'v' (line 49)
        v_461833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 60), 'v', False)
        # Processing the call keyword arguments (line 49)
        kwargs_461834 = {}
        # Getting the type of 'bsr_scale_columns' (line 49)
        bsr_scale_columns_461822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'bsr_scale_columns', False)
        # Calling bsr_scale_columns(args, kwargs) (line 49)
        bsr_scale_columns_call_result_461835 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), bsr_scale_columns_461822, *[int_461823, int_461824, int_461825, int_461826, indptr_461828, indices_461830, data_461832, v_461833], **kwargs_461834)
        
        
        # Call to assert_equal(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to todense(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_461839 = {}
        # Getting the type of 'S' (line 50)
        S_461837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'S', False)
        # Obtaining the member 'todense' of a type (line 50)
        todense_461838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 21), S_461837, 'todense')
        # Calling todense(args, kwargs) (line 50)
        todense_call_result_461840 = invoke(stypy.reporting.localization.Localization(__file__, 50, 21), todense_461838, *[], **kwargs_461839)
        
        # Getting the type of 'E' (line 50)
        E_461841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 34), 'E', False)
        
        # Call to diag(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'v' (line 50)
        v_461843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), 'v', False)
        # Processing the call keyword arguments (line 50)
        kwargs_461844 = {}
        # Getting the type of 'diag' (line 50)
        diag_461842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'diag', False)
        # Calling diag(args, kwargs) (line 50)
        diag_call_result_461845 = invoke(stypy.reporting.localization.Localization(__file__, 50, 36), diag_461842, *[v_461843], **kwargs_461844)
        
        # Applying the binary operator '*' (line 50)
        result_mul_461846 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 34), '*', E_461841, diag_call_result_461845)
        
        # Processing the call keyword arguments (line 50)
        kwargs_461847 = {}
        # Getting the type of 'assert_equal' (line 50)
        assert_equal_461836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 50)
        assert_equal_call_result_461848 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert_equal_461836, *[todense_call_result_461840, result_mul_461846], **kwargs_461847)
        
        
        # ################# End of 'test_scale_rows_and_cols(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scale_rows_and_cols' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_461849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_461849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scale_rows_and_cols'
        return stypy_return_type_461849


    @norecursion
    def test_estimate_blocksize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_estimate_blocksize'
        module_type_store = module_type_store.open_function_context('test_estimate_blocksize', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_localization', localization)
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_function_name', 'TestSparseFunctions.test_estimate_blocksize')
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseFunctions.test_estimate_blocksize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseFunctions.test_estimate_blocksize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_estimate_blocksize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_estimate_blocksize(...)' code ##################

        
        # Assigning a List to a Name (line 53):
        
        # Assigning a List to a Name (line 53):
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_461850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        
        # Assigning a type to the variable 'mats' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'mats', list_461850)
        
        # Call to append(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_461853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_461854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_461855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), list_461854, int_461855)
        # Adding element type (line 54)
        int_461856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), list_461854, int_461856)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), list_461853, list_461854)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_461857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_461858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 27), list_461857, int_461858)
        # Adding element type (line 54)
        int_461859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 27), list_461857, int_461859)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), list_461853, list_461857)
        
        # Processing the call keyword arguments (line 54)
        kwargs_461860 = {}
        # Getting the type of 'mats' (line 54)
        mats_461851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 54)
        append_461852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), mats_461851, 'append')
        # Calling append(args, kwargs) (line 54)
        append_call_result_461861 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), append_461852, *[list_461853], **kwargs_461860)
        
        
        # Call to append(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_461864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_461865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_461866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), list_461865, int_461866)
        # Adding element type (line 55)
        int_461867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), list_461865, int_461867)
        # Adding element type (line 55)
        int_461868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), list_461865, int_461868)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_461864, list_461865)
        # Adding element type (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_461869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_461870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_461869, int_461870)
        # Adding element type (line 55)
        int_461871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_461869, int_461871)
        # Adding element type (line 55)
        int_461872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_461869, int_461872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_461864, list_461869)
        # Adding element type (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_461873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_461874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 37), list_461873, int_461874)
        # Adding element type (line 55)
        int_461875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 37), list_461873, int_461875)
        # Adding element type (line 55)
        int_461876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 37), list_461873, int_461876)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_461864, list_461873)
        
        # Processing the call keyword arguments (line 55)
        kwargs_461877 = {}
        # Getting the type of 'mats' (line 55)
        mats_461862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 55)
        append_461863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), mats_461862, 'append')
        # Calling append(args, kwargs) (line 55)
        append_call_result_461878 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), append_461863, *[list_461864], **kwargs_461877)
        
        
        # Call to append(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_461881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_461882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_461883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 21), list_461882, int_461883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), list_461881, list_461882)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_461884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_461885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_461884, int_461885)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), list_461881, list_461884)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_461886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_461887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 29), list_461886, int_461887)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), list_461881, list_461886)
        
        # Processing the call keyword arguments (line 56)
        kwargs_461888 = {}
        # Getting the type of 'mats' (line 56)
        mats_461879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 56)
        append_461880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), mats_461879, 'append')
        # Calling append(args, kwargs) (line 56)
        append_call_result_461889 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), append_461880, *[list_461881], **kwargs_461888)
        
        
        # Assigning a ListComp to a Name (line 57):
        
        # Assigning a ListComp to a Name (line 57):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'mats' (line 57)
        mats_461894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'mats')
        comprehension_461895 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), mats_461894)
        # Assigning a type to the variable 'x' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'x', comprehension_461895)
        
        # Call to array(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'x' (line 57)
        x_461891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 22), 'x', False)
        # Processing the call keyword arguments (line 57)
        kwargs_461892 = {}
        # Getting the type of 'array' (line 57)
        array_461890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'array', False)
        # Calling array(args, kwargs) (line 57)
        array_call_result_461893 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), array_461890, *[x_461891], **kwargs_461892)
        
        list_461896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), list_461896, array_call_result_461893)
        # Assigning a type to the variable 'mats' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'mats', list_461896)
        
        # Assigning a List to a Name (line 59):
        
        # Assigning a List to a Name (line 59):
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_461897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        
        # Assigning a type to the variable 'blks' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'blks', list_461897)
        
        # Call to append(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_461900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_461901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_461902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 21), list_461901, int_461902)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 20), list_461900, list_461901)
        
        # Processing the call keyword arguments (line 60)
        kwargs_461903 = {}
        # Getting the type of 'blks' (line 60)
        blks_461898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'blks', False)
        # Obtaining the member 'append' of a type (line 60)
        append_461899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), blks_461898, 'append')
        # Calling append(args, kwargs) (line 60)
        append_call_result_461904 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), append_461899, *[list_461900], **kwargs_461903)
        
        
        # Call to append(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_461907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_461908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_461909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 21), list_461908, int_461909)
        # Adding element type (line 61)
        int_461910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 21), list_461908, int_461910)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 20), list_461907, list_461908)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_461911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_461912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 27), list_461911, int_461912)
        # Adding element type (line 61)
        int_461913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 27), list_461911, int_461913)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 20), list_461907, list_461911)
        
        # Processing the call keyword arguments (line 61)
        kwargs_461914 = {}
        # Getting the type of 'blks' (line 61)
        blks_461905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'blks', False)
        # Obtaining the member 'append' of a type (line 61)
        append_461906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), blks_461905, 'append')
        # Calling append(args, kwargs) (line 61)
        append_call_result_461915 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), append_461906, *[list_461907], **kwargs_461914)
        
        
        # Call to append(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_461918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_461919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_461920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 21), list_461919, int_461920)
        # Adding element type (line 62)
        int_461921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 21), list_461919, int_461921)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_461918, list_461919)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_461922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_461923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 27), list_461922, int_461923)
        # Adding element type (line 62)
        int_461924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 27), list_461922, int_461924)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_461918, list_461922)
        
        # Processing the call keyword arguments (line 62)
        kwargs_461925 = {}
        # Getting the type of 'blks' (line 62)
        blks_461916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'blks', False)
        # Obtaining the member 'append' of a type (line 62)
        append_461917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), blks_461916, 'append')
        # Calling append(args, kwargs) (line 62)
        append_call_result_461926 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), append_461917, *[list_461918], **kwargs_461925)
        
        
        # Call to append(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_461929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_461930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        int_461931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_461930, int_461931)
        # Adding element type (line 63)
        int_461932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_461930, int_461932)
        # Adding element type (line 63)
        int_461933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_461930, int_461933)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), list_461929, list_461930)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_461934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        int_461935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), list_461934, int_461935)
        # Adding element type (line 63)
        int_461936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), list_461934, int_461936)
        # Adding element type (line 63)
        int_461937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), list_461934, int_461937)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), list_461929, list_461934)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_461938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        int_461939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 37), list_461938, int_461939)
        # Adding element type (line 63)
        int_461940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 37), list_461938, int_461940)
        # Adding element type (line 63)
        int_461941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 37), list_461938, int_461941)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), list_461929, list_461938)
        
        # Processing the call keyword arguments (line 63)
        kwargs_461942 = {}
        # Getting the type of 'blks' (line 63)
        blks_461927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'blks', False)
        # Obtaining the member 'append' of a type (line 63)
        append_461928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), blks_461927, 'append')
        # Calling append(args, kwargs) (line 63)
        append_call_result_461943 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), append_461928, *[list_461929], **kwargs_461942)
        
        
        # Assigning a ListComp to a Name (line 64):
        
        # Assigning a ListComp to a Name (line 64):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'blks' (line 64)
        blks_461948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'blks')
        comprehension_461949 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), blks_461948)
        # Assigning a type to the variable 'x' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'x', comprehension_461949)
        
        # Call to array(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'x' (line 64)
        x_461945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'x', False)
        # Processing the call keyword arguments (line 64)
        kwargs_461946 = {}
        # Getting the type of 'array' (line 64)
        array_461944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'array', False)
        # Calling array(args, kwargs) (line 64)
        array_call_result_461947 = invoke(stypy.reporting.localization.Localization(__file__, 64, 16), array_461944, *[x_461945], **kwargs_461946)
        
        list_461950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), list_461950, array_call_result_461947)
        # Assigning a type to the variable 'blks' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'blks', list_461950)
        
        # Getting the type of 'mats' (line 66)
        mats_461951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'mats')
        # Testing the type of a for loop iterable (line 66)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 8), mats_461951)
        # Getting the type of the for loop variable (line 66)
        for_loop_var_461952 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 8), mats_461951)
        # Assigning a type to the variable 'A' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'A', for_loop_var_461952)
        # SSA begins for a for statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'blks' (line 67)
        blks_461953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'blks')
        # Testing the type of a for loop iterable (line 67)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 67, 12), blks_461953)
        # Getting the type of the for loop variable (line 67)
        for_loop_var_461954 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 67, 12), blks_461953)
        # Assigning a type to the variable 'B' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'B', for_loop_var_461954)
        # SSA begins for a for statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to kron(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'A' (line 68)
        A_461956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'A', False)
        # Getting the type of 'B' (line 68)
        B_461957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'B', False)
        # Processing the call keyword arguments (line 68)
        kwargs_461958 = {}
        # Getting the type of 'kron' (line 68)
        kron_461955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'kron', False)
        # Calling kron(args, kwargs) (line 68)
        kron_call_result_461959 = invoke(stypy.reporting.localization.Localization(__file__, 68, 20), kron_461955, *[A_461956, B_461957], **kwargs_461958)
        
        # Assigning a type to the variable 'X' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'X', kron_call_result_461959)
        
        # Assigning a Call to a Tuple (line 69):
        
        # Assigning a Subscript to a Name (line 69):
        
        # Obtaining the type of the subscript
        int_461960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 16), 'int')
        
        # Call to estimate_blocksize(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'X' (line 69)
        X_461963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 49), 'X', False)
        # Processing the call keyword arguments (line 69)
        kwargs_461964 = {}
        # Getting the type of 'spfuncs' (line 69)
        spfuncs_461961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'spfuncs', False)
        # Obtaining the member 'estimate_blocksize' of a type (line 69)
        estimate_blocksize_461962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 22), spfuncs_461961, 'estimate_blocksize')
        # Calling estimate_blocksize(args, kwargs) (line 69)
        estimate_blocksize_call_result_461965 = invoke(stypy.reporting.localization.Localization(__file__, 69, 22), estimate_blocksize_461962, *[X_461963], **kwargs_461964)
        
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___461966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), estimate_blocksize_call_result_461965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_461967 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), getitem___461966, int_461960)
        
        # Assigning a type to the variable 'tuple_var_assignment_461520' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'tuple_var_assignment_461520', subscript_call_result_461967)
        
        # Assigning a Subscript to a Name (line 69):
        
        # Obtaining the type of the subscript
        int_461968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 16), 'int')
        
        # Call to estimate_blocksize(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'X' (line 69)
        X_461971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 49), 'X', False)
        # Processing the call keyword arguments (line 69)
        kwargs_461972 = {}
        # Getting the type of 'spfuncs' (line 69)
        spfuncs_461969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'spfuncs', False)
        # Obtaining the member 'estimate_blocksize' of a type (line 69)
        estimate_blocksize_461970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 22), spfuncs_461969, 'estimate_blocksize')
        # Calling estimate_blocksize(args, kwargs) (line 69)
        estimate_blocksize_call_result_461973 = invoke(stypy.reporting.localization.Localization(__file__, 69, 22), estimate_blocksize_461970, *[X_461971], **kwargs_461972)
        
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___461974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), estimate_blocksize_call_result_461973, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_461975 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), getitem___461974, int_461968)
        
        # Assigning a type to the variable 'tuple_var_assignment_461521' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'tuple_var_assignment_461521', subscript_call_result_461975)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'tuple_var_assignment_461520' (line 69)
        tuple_var_assignment_461520_461976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'tuple_var_assignment_461520')
        # Assigning a type to the variable 'r' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'r', tuple_var_assignment_461520_461976)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'tuple_var_assignment_461521' (line 69)
        tuple_var_assignment_461521_461977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'tuple_var_assignment_461521')
        # Assigning a type to the variable 'c' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'c', tuple_var_assignment_461521_461977)
        
        # Call to assert_(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Getting the type of 'r' (line 70)
        r_461979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'r', False)
        
        # Obtaining the type of the subscript
        int_461980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'int')
        # Getting the type of 'B' (line 70)
        B_461981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), 'B', False)
        # Obtaining the member 'shape' of a type (line 70)
        shape_461982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 29), B_461981, 'shape')
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___461983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 29), shape_461982, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_461984 = invoke(stypy.reporting.localization.Localization(__file__, 70, 29), getitem___461983, int_461980)
        
        # Applying the binary operator '>=' (line 70)
        result_ge_461985 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 24), '>=', r_461979, subscript_call_result_461984)
        
        # Processing the call keyword arguments (line 70)
        kwargs_461986 = {}
        # Getting the type of 'assert_' (line 70)
        assert__461978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 70)
        assert__call_result_461987 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), assert__461978, *[result_ge_461985], **kwargs_461986)
        
        
        # Call to assert_(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Getting the type of 'c' (line 71)
        c_461989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'c', False)
        
        # Obtaining the type of the subscript
        int_461990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'int')
        # Getting the type of 'B' (line 71)
        B_461991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 29), 'B', False)
        # Obtaining the member 'shape' of a type (line 71)
        shape_461992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 29), B_461991, 'shape')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___461993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 29), shape_461992, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_461994 = invoke(stypy.reporting.localization.Localization(__file__, 71, 29), getitem___461993, int_461990)
        
        # Applying the binary operator '>=' (line 71)
        result_ge_461995 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 24), '>=', c_461989, subscript_call_result_461994)
        
        # Processing the call keyword arguments (line 71)
        kwargs_461996 = {}
        # Getting the type of 'assert_' (line 71)
        assert__461988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 71)
        assert__call_result_461997 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), assert__461988, *[result_ge_461995], **kwargs_461996)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_estimate_blocksize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_estimate_blocksize' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_461998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_461998)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_estimate_blocksize'
        return stypy_return_type_461998


    @norecursion
    def test_count_blocks(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_count_blocks'
        module_type_store = module_type_store.open_function_context('test_count_blocks', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_localization', localization)
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_function_name', 'TestSparseFunctions.test_count_blocks')
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseFunctions.test_count_blocks.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseFunctions.test_count_blocks', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_count_blocks', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_count_blocks(...)' code ##################


        @norecursion
        def gold(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gold'
            module_type_store = module_type_store.open_function_context('gold', 74, 8, False)
            
            # Passed parameters checking function
            gold.stypy_localization = localization
            gold.stypy_type_of_self = None
            gold.stypy_type_store = module_type_store
            gold.stypy_function_name = 'gold'
            gold.stypy_param_names_list = ['A', 'bs']
            gold.stypy_varargs_param_name = None
            gold.stypy_kwargs_param_name = None
            gold.stypy_call_defaults = defaults
            gold.stypy_call_varargs = varargs
            gold.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gold', ['A', 'bs'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gold', localization, ['A', 'bs'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gold(...)' code ##################

            
            # Assigning a Name to a Tuple (line 75):
            
            # Assigning a Subscript to a Name (line 75):
            
            # Obtaining the type of the subscript
            int_461999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'int')
            # Getting the type of 'bs' (line 75)
            bs_462000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'bs')
            # Obtaining the member '__getitem__' of a type (line 75)
            getitem___462001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), bs_462000, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
            subscript_call_result_462002 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), getitem___462001, int_461999)
            
            # Assigning a type to the variable 'tuple_var_assignment_461522' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'tuple_var_assignment_461522', subscript_call_result_462002)
            
            # Assigning a Subscript to a Name (line 75):
            
            # Obtaining the type of the subscript
            int_462003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'int')
            # Getting the type of 'bs' (line 75)
            bs_462004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'bs')
            # Obtaining the member '__getitem__' of a type (line 75)
            getitem___462005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), bs_462004, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
            subscript_call_result_462006 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), getitem___462005, int_462003)
            
            # Assigning a type to the variable 'tuple_var_assignment_461523' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'tuple_var_assignment_461523', subscript_call_result_462006)
            
            # Assigning a Name to a Name (line 75):
            # Getting the type of 'tuple_var_assignment_461522' (line 75)
            tuple_var_assignment_461522_462007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'tuple_var_assignment_461522')
            # Assigning a type to the variable 'R' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'R', tuple_var_assignment_461522_462007)
            
            # Assigning a Name to a Name (line 75):
            # Getting the type of 'tuple_var_assignment_461523' (line 75)
            tuple_var_assignment_461523_462008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'tuple_var_assignment_461523')
            # Assigning a type to the variable 'C' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'C', tuple_var_assignment_461523_462008)
            
            # Assigning a Call to a Tuple (line 76):
            
            # Assigning a Subscript to a Name (line 76):
            
            # Obtaining the type of the subscript
            int_462009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'int')
            
            # Call to nonzero(...): (line 76)
            # Processing the call keyword arguments (line 76)
            kwargs_462012 = {}
            # Getting the type of 'A' (line 76)
            A_462010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'A', False)
            # Obtaining the member 'nonzero' of a type (line 76)
            nonzero_462011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 18), A_462010, 'nonzero')
            # Calling nonzero(args, kwargs) (line 76)
            nonzero_call_result_462013 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), nonzero_462011, *[], **kwargs_462012)
            
            # Obtaining the member '__getitem__' of a type (line 76)
            getitem___462014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), nonzero_call_result_462013, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 76)
            subscript_call_result_462015 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), getitem___462014, int_462009)
            
            # Assigning a type to the variable 'tuple_var_assignment_461524' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_var_assignment_461524', subscript_call_result_462015)
            
            # Assigning a Subscript to a Name (line 76):
            
            # Obtaining the type of the subscript
            int_462016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'int')
            
            # Call to nonzero(...): (line 76)
            # Processing the call keyword arguments (line 76)
            kwargs_462019 = {}
            # Getting the type of 'A' (line 76)
            A_462017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'A', False)
            # Obtaining the member 'nonzero' of a type (line 76)
            nonzero_462018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 18), A_462017, 'nonzero')
            # Calling nonzero(args, kwargs) (line 76)
            nonzero_call_result_462020 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), nonzero_462018, *[], **kwargs_462019)
            
            # Obtaining the member '__getitem__' of a type (line 76)
            getitem___462021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), nonzero_call_result_462020, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 76)
            subscript_call_result_462022 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), getitem___462021, int_462016)
            
            # Assigning a type to the variable 'tuple_var_assignment_461525' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_var_assignment_461525', subscript_call_result_462022)
            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'tuple_var_assignment_461524' (line 76)
            tuple_var_assignment_461524_462023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_var_assignment_461524')
            # Assigning a type to the variable 'I' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'I', tuple_var_assignment_461524_462023)
            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'tuple_var_assignment_461525' (line 76)
            tuple_var_assignment_461525_462024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_var_assignment_461525')
            # Assigning a type to the variable 'J' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'J', tuple_var_assignment_461525_462024)
            
            # Call to len(...): (line 77)
            # Processing the call arguments (line 77)
            
            # Call to set(...): (line 77)
            # Processing the call arguments (line 77)
            
            # Call to zip(...): (line 77)
            # Processing the call arguments (line 77)
            # Getting the type of 'I' (line 77)
            I_462028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'I', False)
            # Getting the type of 'R' (line 77)
            R_462029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 'R', False)
            # Applying the binary operator '//' (line 77)
            result_floordiv_462030 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 31), '//', I_462028, R_462029)
            
            # Getting the type of 'J' (line 77)
            J_462031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'J', False)
            # Getting the type of 'C' (line 77)
            C_462032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 39), 'C', False)
            # Applying the binary operator '//' (line 77)
            result_floordiv_462033 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 36), '//', J_462031, C_462032)
            
            # Processing the call keyword arguments (line 77)
            kwargs_462034 = {}
            # Getting the type of 'zip' (line 77)
            zip_462027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'zip', False)
            # Calling zip(args, kwargs) (line 77)
            zip_call_result_462035 = invoke(stypy.reporting.localization.Localization(__file__, 77, 27), zip_462027, *[result_floordiv_462030, result_floordiv_462033], **kwargs_462034)
            
            # Processing the call keyword arguments (line 77)
            kwargs_462036 = {}
            # Getting the type of 'set' (line 77)
            set_462026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'set', False)
            # Calling set(args, kwargs) (line 77)
            set_call_result_462037 = invoke(stypy.reporting.localization.Localization(__file__, 77, 23), set_462026, *[zip_call_result_462035], **kwargs_462036)
            
            # Processing the call keyword arguments (line 77)
            kwargs_462038 = {}
            # Getting the type of 'len' (line 77)
            len_462025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'len', False)
            # Calling len(args, kwargs) (line 77)
            len_call_result_462039 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), len_462025, *[set_call_result_462037], **kwargs_462038)
            
            # Assigning a type to the variable 'stypy_return_type' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type', len_call_result_462039)
            
            # ################# End of 'gold(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gold' in the type store
            # Getting the type of 'stypy_return_type' (line 74)
            stypy_return_type_462040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_462040)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gold'
            return stypy_return_type_462040

        # Assigning a type to the variable 'gold' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'gold', gold)
        
        # Assigning a List to a Name (line 79):
        
        # Assigning a List to a Name (line 79):
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_462041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        
        # Assigning a type to the variable 'mats' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'mats', list_462041)
        
        # Call to append(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_462044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_462045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_462046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_462045, int_462046)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 20), list_462044, list_462045)
        
        # Processing the call keyword arguments (line 80)
        kwargs_462047 = {}
        # Getting the type of 'mats' (line 80)
        mats_462042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 80)
        append_462043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), mats_462042, 'append')
        # Calling append(args, kwargs) (line 80)
        append_call_result_462048 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), append_462043, *[list_462044], **kwargs_462047)
        
        
        # Call to append(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_462051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_462052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        int_462053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_462052, int_462053)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), list_462051, list_462052)
        
        # Processing the call keyword arguments (line 81)
        kwargs_462054 = {}
        # Getting the type of 'mats' (line 81)
        mats_462049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 81)
        append_462050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), mats_462049, 'append')
        # Calling append(args, kwargs) (line 81)
        append_call_result_462055 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), append_462050, *[list_462051], **kwargs_462054)
        
        
        # Call to append(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_462058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_462059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        int_462060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 21), list_462059, int_462060)
        # Adding element type (line 82)
        int_462061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 21), list_462059, int_462061)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), list_462058, list_462059)
        
        # Processing the call keyword arguments (line 82)
        kwargs_462062 = {}
        # Getting the type of 'mats' (line 82)
        mats_462056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 82)
        append_462057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), mats_462056, 'append')
        # Calling append(args, kwargs) (line 82)
        append_call_result_462063 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), append_462057, *[list_462058], **kwargs_462062)
        
        
        # Call to append(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_462066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_462067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        int_462068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_462067, int_462068)
        # Adding element type (line 83)
        int_462069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_462067, int_462069)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), list_462066, list_462067)
        
        # Processing the call keyword arguments (line 83)
        kwargs_462070 = {}
        # Getting the type of 'mats' (line 83)
        mats_462064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 83)
        append_462065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), mats_462064, 'append')
        # Calling append(args, kwargs) (line 83)
        append_call_result_462071 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), append_462065, *[list_462066], **kwargs_462070)
        
        
        # Call to append(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_462074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_462075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_462076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), list_462075, int_462076)
        # Adding element type (line 84)
        int_462077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), list_462075, int_462077)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), list_462074, list_462075)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_462078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_462079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 27), list_462078, int_462079)
        # Adding element type (line 84)
        int_462080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 27), list_462078, int_462080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), list_462074, list_462078)
        
        # Processing the call keyword arguments (line 84)
        kwargs_462081 = {}
        # Getting the type of 'mats' (line 84)
        mats_462072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 84)
        append_462073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), mats_462072, 'append')
        # Calling append(args, kwargs) (line 84)
        append_call_result_462082 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), append_462073, *[list_462074], **kwargs_462081)
        
        
        # Call to append(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_462085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_462086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        int_462087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_462086, int_462087)
        # Adding element type (line 85)
        int_462088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_462086, int_462088)
        # Adding element type (line 85)
        int_462089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_462086, int_462089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 20), list_462085, list_462086)
        # Adding element type (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_462090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        int_462091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 29), list_462090, int_462091)
        # Adding element type (line 85)
        int_462092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 29), list_462090, int_462092)
        # Adding element type (line 85)
        int_462093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 29), list_462090, int_462093)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 20), list_462085, list_462090)
        # Adding element type (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_462094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        int_462095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 37), list_462094, int_462095)
        # Adding element type (line 85)
        int_462096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 37), list_462094, int_462096)
        # Adding element type (line 85)
        int_462097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 37), list_462094, int_462097)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 20), list_462085, list_462094)
        
        # Processing the call keyword arguments (line 85)
        kwargs_462098 = {}
        # Getting the type of 'mats' (line 85)
        mats_462083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 85)
        append_462084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), mats_462083, 'append')
        # Calling append(args, kwargs) (line 85)
        append_call_result_462099 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), append_462084, *[list_462085], **kwargs_462098)
        
        
        # Call to append(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_462102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_462103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        int_462104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), list_462103, int_462104)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), list_462102, list_462103)
        # Adding element type (line 86)
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_462105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        int_462106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), list_462105, int_462106)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), list_462102, list_462105)
        # Adding element type (line 86)
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_462107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        int_462108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 29), list_462107, int_462108)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), list_462102, list_462107)
        
        # Processing the call keyword arguments (line 86)
        kwargs_462109 = {}
        # Getting the type of 'mats' (line 86)
        mats_462100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 86)
        append_462101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), mats_462100, 'append')
        # Calling append(args, kwargs) (line 86)
        append_call_result_462110 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), append_462101, *[list_462102], **kwargs_462109)
        
        
        # Getting the type of 'mats' (line 88)
        mats_462111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'mats')
        # Testing the type of a for loop iterable (line 88)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), mats_462111)
        # Getting the type of the for loop variable (line 88)
        for_loop_var_462112 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), mats_462111)
        # Assigning a type to the variable 'A' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'A', for_loop_var_462112)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'mats' (line 89)
        mats_462113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'mats')
        # Testing the type of a for loop iterable (line 89)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 12), mats_462113)
        # Getting the type of the for loop variable (line 89)
        for_loop_var_462114 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 12), mats_462113)
        # Assigning a type to the variable 'B' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'B', for_loop_var_462114)
        # SSA begins for a for statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to kron(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'A' (line 90)
        A_462116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'A', False)
        # Getting the type of 'B' (line 90)
        B_462117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'B', False)
        # Processing the call keyword arguments (line 90)
        kwargs_462118 = {}
        # Getting the type of 'kron' (line 90)
        kron_462115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'kron', False)
        # Calling kron(args, kwargs) (line 90)
        kron_call_result_462119 = invoke(stypy.reporting.localization.Localization(__file__, 90, 20), kron_462115, *[A_462116, B_462117], **kwargs_462118)
        
        # Assigning a type to the variable 'X' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'X', kron_call_result_462119)
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to csr_matrix(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'X' (line 91)
        X_462121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'X', False)
        # Processing the call keyword arguments (line 91)
        kwargs_462122 = {}
        # Getting the type of 'csr_matrix' (line 91)
        csr_matrix_462120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 91)
        csr_matrix_call_result_462123 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), csr_matrix_462120, *[X_462121], **kwargs_462122)
        
        # Assigning a type to the variable 'Y' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'Y', csr_matrix_call_result_462123)
        
        
        # Call to range(...): (line 92)
        # Processing the call arguments (line 92)
        int_462125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 31), 'int')
        int_462126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 33), 'int')
        # Processing the call keyword arguments (line 92)
        kwargs_462127 = {}
        # Getting the type of 'range' (line 92)
        range_462124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'range', False)
        # Calling range(args, kwargs) (line 92)
        range_call_result_462128 = invoke(stypy.reporting.localization.Localization(__file__, 92, 25), range_462124, *[int_462125, int_462126], **kwargs_462127)
        
        # Testing the type of a for loop iterable (line 92)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 92, 16), range_call_result_462128)
        # Getting the type of the for loop variable (line 92)
        for_loop_var_462129 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 92, 16), range_call_result_462128)
        # Assigning a type to the variable 'R' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'R', for_loop_var_462129)
        # SSA begins for a for statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 93)
        # Processing the call arguments (line 93)
        int_462131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 35), 'int')
        int_462132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 37), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_462133 = {}
        # Getting the type of 'range' (line 93)
        range_462130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 29), 'range', False)
        # Calling range(args, kwargs) (line 93)
        range_call_result_462134 = invoke(stypy.reporting.localization.Localization(__file__, 93, 29), range_462130, *[int_462131, int_462132], **kwargs_462133)
        
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 20), range_call_result_462134)
        # Getting the type of the for loop variable (line 93)
        for_loop_var_462135 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 20), range_call_result_462134)
        # Assigning a type to the variable 'C' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'C', for_loop_var_462135)
        # SSA begins for a for statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to count_blocks(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'Y' (line 94)
        Y_462139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 58), 'Y', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 94)
        tuple_462140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 94)
        # Adding element type (line 94)
        # Getting the type of 'R' (line 94)
        R_462141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 62), 'R', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 62), tuple_462140, R_462141)
        # Adding element type (line 94)
        # Getting the type of 'C' (line 94)
        C_462142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 65), 'C', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 62), tuple_462140, C_462142)
        
        # Processing the call keyword arguments (line 94)
        kwargs_462143 = {}
        # Getting the type of 'spfuncs' (line 94)
        spfuncs_462137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 37), 'spfuncs', False)
        # Obtaining the member 'count_blocks' of a type (line 94)
        count_blocks_462138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 37), spfuncs_462137, 'count_blocks')
        # Calling count_blocks(args, kwargs) (line 94)
        count_blocks_call_result_462144 = invoke(stypy.reporting.localization.Localization(__file__, 94, 37), count_blocks_462138, *[Y_462139, tuple_462140], **kwargs_462143)
        
        
        # Call to gold(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'X' (line 94)
        X_462146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 75), 'X', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 94)
        tuple_462147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 79), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 94)
        # Adding element type (line 94)
        # Getting the type of 'R' (line 94)
        R_462148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 79), 'R', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 79), tuple_462147, R_462148)
        # Adding element type (line 94)
        # Getting the type of 'C' (line 94)
        C_462149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 82), 'C', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 79), tuple_462147, C_462149)
        
        # Processing the call keyword arguments (line 94)
        kwargs_462150 = {}
        # Getting the type of 'gold' (line 94)
        gold_462145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 70), 'gold', False)
        # Calling gold(args, kwargs) (line 94)
        gold_call_result_462151 = invoke(stypy.reporting.localization.Localization(__file__, 94, 70), gold_462145, *[X_462146, tuple_462147], **kwargs_462150)
        
        # Processing the call keyword arguments (line 94)
        kwargs_462152 = {}
        # Getting the type of 'assert_equal' (line 94)
        assert_equal_462136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 94)
        assert_equal_call_result_462153 = invoke(stypy.reporting.localization.Localization(__file__, 94, 24), assert_equal_462136, *[count_blocks_call_result_462144, gold_call_result_462151], **kwargs_462152)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to kron(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_462155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_462156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_462157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 18), list_462156, int_462157)
        # Adding element type (line 96)
        int_462158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 18), list_462156, int_462158)
        # Adding element type (line 96)
        int_462159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 18), list_462156, int_462159)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_462155, list_462156)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_462160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_462161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 26), list_462160, int_462161)
        # Adding element type (line 96)
        int_462162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 26), list_462160, int_462162)
        # Adding element type (line 96)
        int_462163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 26), list_462160, int_462163)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_462155, list_462160)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_462164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_462165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 34), list_462164, int_462165)
        # Adding element type (line 96)
        int_462166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 34), list_462164, int_462166)
        # Adding element type (line 96)
        int_462167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 34), list_462164, int_462167)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_462155, list_462164)
        
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_462168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_462169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_462170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 44), list_462169, int_462170)
        # Adding element type (line 96)
        int_462171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 44), list_462169, int_462171)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 43), list_462168, list_462169)
        
        # Processing the call keyword arguments (line 96)
        kwargs_462172 = {}
        # Getting the type of 'kron' (line 96)
        kron_462154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'kron', False)
        # Calling kron(args, kwargs) (line 96)
        kron_call_result_462173 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), kron_462154, *[list_462155, list_462168], **kwargs_462172)
        
        # Assigning a type to the variable 'X' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'X', kron_call_result_462173)
        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to csc_matrix(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'X' (line 97)
        X_462175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'X', False)
        # Processing the call keyword arguments (line 97)
        kwargs_462176 = {}
        # Getting the type of 'csc_matrix' (line 97)
        csc_matrix_462174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 97)
        csc_matrix_call_result_462177 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), csc_matrix_462174, *[X_462175], **kwargs_462176)
        
        # Assigning a type to the variable 'Y' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'Y', csc_matrix_call_result_462177)
        
        # Call to assert_equal(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to count_blocks(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'X' (line 98)
        X_462181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'X', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 98)
        tuple_462182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 98)
        # Adding element type (line 98)
        int_462183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 46), tuple_462182, int_462183)
        # Adding element type (line 98)
        int_462184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 46), tuple_462182, int_462184)
        
        # Processing the call keyword arguments (line 98)
        kwargs_462185 = {}
        # Getting the type of 'spfuncs' (line 98)
        spfuncs_462179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'spfuncs', False)
        # Obtaining the member 'count_blocks' of a type (line 98)
        count_blocks_462180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 21), spfuncs_462179, 'count_blocks')
        # Calling count_blocks(args, kwargs) (line 98)
        count_blocks_call_result_462186 = invoke(stypy.reporting.localization.Localization(__file__, 98, 21), count_blocks_462180, *[X_462181, tuple_462182], **kwargs_462185)
        
        
        # Call to gold(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'X' (line 98)
        X_462188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 59), 'X', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 98)
        tuple_462189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 98)
        # Adding element type (line 98)
        int_462190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 63), tuple_462189, int_462190)
        # Adding element type (line 98)
        int_462191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 63), tuple_462189, int_462191)
        
        # Processing the call keyword arguments (line 98)
        kwargs_462192 = {}
        # Getting the type of 'gold' (line 98)
        gold_462187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 54), 'gold', False)
        # Calling gold(args, kwargs) (line 98)
        gold_call_result_462193 = invoke(stypy.reporting.localization.Localization(__file__, 98, 54), gold_462187, *[X_462188, tuple_462189], **kwargs_462192)
        
        # Processing the call keyword arguments (line 98)
        kwargs_462194 = {}
        # Getting the type of 'assert_equal' (line 98)
        assert_equal_462178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 98)
        assert_equal_call_result_462195 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), assert_equal_462178, *[count_blocks_call_result_462186, gold_call_result_462193], **kwargs_462194)
        
        
        # Call to assert_equal(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to count_blocks(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'Y' (line 99)
        Y_462199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 42), 'Y', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_462200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        int_462201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 46), tuple_462200, int_462201)
        # Adding element type (line 99)
        int_462202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 46), tuple_462200, int_462202)
        
        # Processing the call keyword arguments (line 99)
        kwargs_462203 = {}
        # Getting the type of 'spfuncs' (line 99)
        spfuncs_462197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'spfuncs', False)
        # Obtaining the member 'count_blocks' of a type (line 99)
        count_blocks_462198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 21), spfuncs_462197, 'count_blocks')
        # Calling count_blocks(args, kwargs) (line 99)
        count_blocks_call_result_462204 = invoke(stypy.reporting.localization.Localization(__file__, 99, 21), count_blocks_462198, *[Y_462199, tuple_462200], **kwargs_462203)
        
        
        # Call to gold(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'X' (line 99)
        X_462206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 59), 'X', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_462207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        int_462208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 63), tuple_462207, int_462208)
        # Adding element type (line 99)
        int_462209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 63), tuple_462207, int_462209)
        
        # Processing the call keyword arguments (line 99)
        kwargs_462210 = {}
        # Getting the type of 'gold' (line 99)
        gold_462205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 54), 'gold', False)
        # Calling gold(args, kwargs) (line 99)
        gold_call_result_462211 = invoke(stypy.reporting.localization.Localization(__file__, 99, 54), gold_462205, *[X_462206, tuple_462207], **kwargs_462210)
        
        # Processing the call keyword arguments (line 99)
        kwargs_462212 = {}
        # Getting the type of 'assert_equal' (line 99)
        assert_equal_462196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 99)
        assert_equal_call_result_462213 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assert_equal_462196, *[count_blocks_call_result_462204, gold_call_result_462211], **kwargs_462212)
        
        
        # ################# End of 'test_count_blocks(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_count_blocks' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_462214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_count_blocks'
        return stypy_return_type_462214


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseFunctions.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSparseFunctions' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestSparseFunctions', TestSparseFunctions)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
