
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test functions for the sparse.linalg.norm module
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.linalg import norm as npnorm
8: from numpy.testing import assert_equal, assert_allclose
9: from pytest import raises as assert_raises
10: 
11: from scipy._lib._version import NumpyVersion
12: import scipy.sparse
13: from scipy.sparse.linalg import norm as spnorm
14: 
15: 
16: class TestNorm(object):
17:     def setup_method(self):
18:         a = np.arange(9) - 4
19:         b = a.reshape((3, 3))
20:         self.b = scipy.sparse.csr_matrix(b)
21: 
22:     def test_matrix_norm(self):
23: 
24:         # Frobenius norm is the default
25:         assert_allclose(spnorm(self.b), 7.745966692414834)        
26:         assert_allclose(spnorm(self.b, 'fro'), 7.745966692414834)
27: 
28:         assert_allclose(spnorm(self.b, np.inf), 9)
29:         assert_allclose(spnorm(self.b, -np.inf), 2)
30:         assert_allclose(spnorm(self.b, 1), 7)
31:         assert_allclose(spnorm(self.b, -1), 6)
32: 
33:         # _multi_svd_norm is not implemented for sparse matrix
34:         assert_raises(NotImplementedError, spnorm, self.b, 2)
35:         assert_raises(NotImplementedError, spnorm, self.b, -2)
36: 
37:     def test_matrix_norm_axis(self):
38:         for m, axis in ((self.b, None), (self.b, (0, 1)), (self.b.T, (1, 0))):
39:             assert_allclose(spnorm(m, axis=axis), 7.745966692414834)        
40:             assert_allclose(spnorm(m, 'fro', axis=axis), 7.745966692414834)
41:             assert_allclose(spnorm(m, np.inf, axis=axis), 9)
42:             assert_allclose(spnorm(m, -np.inf, axis=axis), 2)
43:             assert_allclose(spnorm(m, 1, axis=axis), 7)
44:             assert_allclose(spnorm(m, -1, axis=axis), 6)
45: 
46:     def test_vector_norm(self):
47:         v = [4.5825756949558398, 4.2426406871192848, 4.5825756949558398]
48:         for m, a in (self.b, 0), (self.b.T, 1):
49:             for axis in a, (a, ), a-2, (a-2, ):
50:                 assert_allclose(spnorm(m, 1, axis=axis), [7, 6, 7])
51:                 assert_allclose(spnorm(m, np.inf, axis=axis), [4, 3, 4])
52:                 assert_allclose(spnorm(m, axis=axis), v)
53:                 assert_allclose(spnorm(m, ord=2, axis=axis), v)
54:                 assert_allclose(spnorm(m, ord=None, axis=axis), v)
55: 
56:     def test_norm_exceptions(self):
57:         m = self.b
58:         assert_raises(TypeError, spnorm, m, None, 1.5)
59:         assert_raises(TypeError, spnorm, m, None, [2])
60:         assert_raises(ValueError, spnorm, m, None, ())
61:         assert_raises(ValueError, spnorm, m, None, (0, 1, 2))
62:         assert_raises(ValueError, spnorm, m, None, (0, 0))
63:         assert_raises(ValueError, spnorm, m, None, (0, 2))
64:         assert_raises(ValueError, spnorm, m, None, (-3, 0))
65:         assert_raises(ValueError, spnorm, m, None, 2)
66:         assert_raises(ValueError, spnorm, m, None, -3)
67:         assert_raises(ValueError, spnorm, m, 'plate_of_shrimp', 0)
68:         assert_raises(ValueError, spnorm, m, 'plate_of_shrimp', (0, 1))
69: 
70: 
71: class TestVsNumpyNorm(object):
72:     _sparse_types = (
73:             scipy.sparse.bsr_matrix,
74:             scipy.sparse.coo_matrix,
75:             scipy.sparse.csc_matrix,
76:             scipy.sparse.csr_matrix,
77:             scipy.sparse.dia_matrix,
78:             scipy.sparse.dok_matrix,
79:             scipy.sparse.lil_matrix,
80:             )
81:     _test_matrices = (
82:             (np.arange(9) - 4).reshape((3, 3)),
83:             [
84:                 [1, 2, 3],
85:                 [-1, 1, 4]],
86:             [
87:                 [1, 0, 3],
88:                 [-1, 1, 4j]],
89:             )
90: 
91:     def test_sparse_matrix_norms(self):
92:         for sparse_type in self._sparse_types:
93:             for M in self._test_matrices:
94:                 S = sparse_type(M)
95:                 assert_allclose(spnorm(S), npnorm(M))
96:                 assert_allclose(spnorm(S, 'fro'), npnorm(M, 'fro'))
97:                 assert_allclose(spnorm(S, np.inf), npnorm(M, np.inf))
98:                 assert_allclose(spnorm(S, -np.inf), npnorm(M, -np.inf))
99:                 assert_allclose(spnorm(S, 1), npnorm(M, 1))
100:                 assert_allclose(spnorm(S, -1), npnorm(M, -1))
101: 
102:     def test_sparse_matrix_norms_with_axis(self):
103:         for sparse_type in self._sparse_types:
104:             for M in self._test_matrices:
105:                 S = sparse_type(M)
106:                 for axis in None, (0, 1), (1, 0):
107:                     assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
108:                     for ord in 'fro', np.inf, -np.inf, 1, -1:
109:                         assert_allclose(spnorm(S, ord, axis=axis),
110:                                         npnorm(M, ord, axis=axis))
111:                 # Some numpy matrix norms are allergic to negative axes.
112:                 for axis in (-2, -1), (-1, -2), (1, -2):
113:                     assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
114:                     assert_allclose(spnorm(S, 'f', axis=axis),
115:                                     npnorm(M, 'f', axis=axis))
116:                     assert_allclose(spnorm(S, 'fro', axis=axis),
117:                                     npnorm(M, 'fro', axis=axis))
118: 
119:     def test_sparse_vector_norms(self):
120:         for sparse_type in self._sparse_types:
121:             for M in self._test_matrices:
122:                 S = sparse_type(M)
123:                 for axis in (0, 1, -1, -2, (0, ), (1, ), (-1, ), (-2, )):
124:                     assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
125:                     for ord in None, 2, np.inf, -np.inf, 1, 0.5, 0.42:
126:                         assert_allclose(spnorm(S, ord, axis=axis),
127:                                         npnorm(M, ord, axis=axis))
128: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_427958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Test functions for the sparse.linalg.norm module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_427959 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_427959) is not StypyTypeError):

    if (import_427959 != 'pyd_module'):
        __import__(import_427959)
        sys_modules_427960 = sys.modules[import_427959]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_427960.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_427959)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.linalg import npnorm' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_427961 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg')

if (type(import_427961) is not StypyTypeError):

    if (import_427961 != 'pyd_module'):
        __import__(import_427961)
        sys_modules_427962 = sys.modules[import_427961]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', sys_modules_427962.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_427962, sys_modules_427962.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm as npnorm

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', None, module_type_store, ['norm'], [npnorm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', import_427961)

# Adding an alias
module_type_store.add_alias('npnorm', 'norm')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_equal, assert_allclose' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_427963 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_427963) is not StypyTypeError):

    if (import_427963 != 'pyd_module'):
        __import__(import_427963)
        sys_modules_427964 = sys.modules[import_427963]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_427964.module_type_store, module_type_store, ['assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_427964, sys_modules_427964.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose'], [assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_427963)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from pytest import assert_raises' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_427965 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_427965) is not StypyTypeError):

    if (import_427965 != 'pyd_module'):
        __import__(import_427965)
        sys_modules_427966 = sys.modules[import_427965]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_427966.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_427966, sys_modules_427966.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_427965)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._version import NumpyVersion' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_427967 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._version')

if (type(import_427967) is not StypyTypeError):

    if (import_427967 != 'pyd_module'):
        __import__(import_427967)
        sys_modules_427968 = sys.modules[import_427967]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._version', sys_modules_427968.module_type_store, module_type_store, ['NumpyVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_427968, sys_modules_427968.module_type_store, module_type_store)
    else:
        from scipy._lib._version import NumpyVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._version', None, module_type_store, ['NumpyVersion'], [NumpyVersion])

else:
    # Assigning a type to the variable 'scipy._lib._version' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._version', import_427967)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import scipy.sparse' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_427969 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse')

if (type(import_427969) is not StypyTypeError):

    if (import_427969 != 'pyd_module'):
        __import__(import_427969)
        sys_modules_427970 = sys.modules[import_427969]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', sys_modules_427970.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', import_427969)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.linalg import spnorm' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_427971 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg')

if (type(import_427971) is not StypyTypeError):

    if (import_427971 != 'pyd_module'):
        __import__(import_427971)
        sys_modules_427972 = sys.modules[import_427971]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg', sys_modules_427972.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_427972, sys_modules_427972.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import norm as spnorm

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg', None, module_type_store, ['norm'], [spnorm])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg', import_427971)

# Adding an alias
module_type_store.add_alias('spnorm', 'norm')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

# Declaration of the 'TestNorm' class

class TestNorm(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNorm.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestNorm.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNorm.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNorm.setup_method.__dict__.__setitem__('stypy_function_name', 'TestNorm.setup_method')
        TestNorm.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestNorm.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNorm.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNorm.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNorm.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNorm.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNorm.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNorm.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a BinOp to a Name (line 18):
        
        # Call to arange(...): (line 18)
        # Processing the call arguments (line 18)
        int_427975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'int')
        # Processing the call keyword arguments (line 18)
        kwargs_427976 = {}
        # Getting the type of 'np' (line 18)
        np_427973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 18)
        arange_427974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 12), np_427973, 'arange')
        # Calling arange(args, kwargs) (line 18)
        arange_call_result_427977 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), arange_427974, *[int_427975], **kwargs_427976)
        
        int_427978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'int')
        # Applying the binary operator '-' (line 18)
        result_sub_427979 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 12), '-', arange_call_result_427977, int_427978)
        
        # Assigning a type to the variable 'a' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'a', result_sub_427979)
        
        # Assigning a Call to a Name (line 19):
        
        # Call to reshape(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 19)
        tuple_427982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        # Adding element type (line 19)
        int_427983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), tuple_427982, int_427983)
        # Adding element type (line 19)
        int_427984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), tuple_427982, int_427984)
        
        # Processing the call keyword arguments (line 19)
        kwargs_427985 = {}
        # Getting the type of 'a' (line 19)
        a_427980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'a', False)
        # Obtaining the member 'reshape' of a type (line 19)
        reshape_427981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), a_427980, 'reshape')
        # Calling reshape(args, kwargs) (line 19)
        reshape_call_result_427986 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), reshape_427981, *[tuple_427982], **kwargs_427985)
        
        # Assigning a type to the variable 'b' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'b', reshape_call_result_427986)
        
        # Assigning a Call to a Attribute (line 20):
        
        # Call to csr_matrix(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'b' (line 20)
        b_427990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 41), 'b', False)
        # Processing the call keyword arguments (line 20)
        kwargs_427991 = {}
        # Getting the type of 'scipy' (line 20)
        scipy_427987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 20)
        sparse_427988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 17), scipy_427987, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 20)
        csr_matrix_427989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 17), sparse_427988, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 20)
        csr_matrix_call_result_427992 = invoke(stypy.reporting.localization.Localization(__file__, 20, 17), csr_matrix_427989, *[b_427990], **kwargs_427991)
        
        # Getting the type of 'self' (line 20)
        self_427993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'b' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_427993, 'b', csr_matrix_call_result_427992)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_427994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427994)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_427994


    @norecursion
    def test_matrix_norm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matrix_norm'
        module_type_store = module_type_store.open_function_context('test_matrix_norm', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_localization', localization)
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_function_name', 'TestNorm.test_matrix_norm')
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_param_names_list', [])
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNorm.test_matrix_norm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNorm.test_matrix_norm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matrix_norm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matrix_norm(...)' code ##################

        
        # Call to assert_allclose(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Call to spnorm(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'self' (line 25)
        self_427997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'self', False)
        # Obtaining the member 'b' of a type (line 25)
        b_427998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 31), self_427997, 'b')
        # Processing the call keyword arguments (line 25)
        kwargs_427999 = {}
        # Getting the type of 'spnorm' (line 25)
        spnorm_427996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 25)
        spnorm_call_result_428000 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), spnorm_427996, *[b_427998], **kwargs_427999)
        
        float_428001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 40), 'float')
        # Processing the call keyword arguments (line 25)
        kwargs_428002 = {}
        # Getting the type of 'assert_allclose' (line 25)
        assert_allclose_427995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 25)
        assert_allclose_call_result_428003 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assert_allclose_427995, *[spnorm_call_result_428000, float_428001], **kwargs_428002)
        
        
        # Call to assert_allclose(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to spnorm(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'self' (line 26)
        self_428006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'self', False)
        # Obtaining the member 'b' of a type (line 26)
        b_428007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 31), self_428006, 'b')
        str_428008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'str', 'fro')
        # Processing the call keyword arguments (line 26)
        kwargs_428009 = {}
        # Getting the type of 'spnorm' (line 26)
        spnorm_428005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 26)
        spnorm_call_result_428010 = invoke(stypy.reporting.localization.Localization(__file__, 26, 24), spnorm_428005, *[b_428007, str_428008], **kwargs_428009)
        
        float_428011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 47), 'float')
        # Processing the call keyword arguments (line 26)
        kwargs_428012 = {}
        # Getting the type of 'assert_allclose' (line 26)
        assert_allclose_428004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 26)
        assert_allclose_call_result_428013 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assert_allclose_428004, *[spnorm_call_result_428010, float_428011], **kwargs_428012)
        
        
        # Call to assert_allclose(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Call to spnorm(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_428016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'self', False)
        # Obtaining the member 'b' of a type (line 28)
        b_428017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 31), self_428016, 'b')
        # Getting the type of 'np' (line 28)
        np_428018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 39), 'np', False)
        # Obtaining the member 'inf' of a type (line 28)
        inf_428019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 39), np_428018, 'inf')
        # Processing the call keyword arguments (line 28)
        kwargs_428020 = {}
        # Getting the type of 'spnorm' (line 28)
        spnorm_428015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 28)
        spnorm_call_result_428021 = invoke(stypy.reporting.localization.Localization(__file__, 28, 24), spnorm_428015, *[b_428017, inf_428019], **kwargs_428020)
        
        int_428022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 48), 'int')
        # Processing the call keyword arguments (line 28)
        kwargs_428023 = {}
        # Getting the type of 'assert_allclose' (line 28)
        assert_allclose_428014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 28)
        assert_allclose_call_result_428024 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assert_allclose_428014, *[spnorm_call_result_428021, int_428022], **kwargs_428023)
        
        
        # Call to assert_allclose(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Call to spnorm(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'self' (line 29)
        self_428027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'self', False)
        # Obtaining the member 'b' of a type (line 29)
        b_428028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 31), self_428027, 'b')
        
        # Getting the type of 'np' (line 29)
        np_428029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 40), 'np', False)
        # Obtaining the member 'inf' of a type (line 29)
        inf_428030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 40), np_428029, 'inf')
        # Applying the 'usub' unary operator (line 29)
        result___neg___428031 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 39), 'usub', inf_428030)
        
        # Processing the call keyword arguments (line 29)
        kwargs_428032 = {}
        # Getting the type of 'spnorm' (line 29)
        spnorm_428026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 29)
        spnorm_call_result_428033 = invoke(stypy.reporting.localization.Localization(__file__, 29, 24), spnorm_428026, *[b_428028, result___neg___428031], **kwargs_428032)
        
        int_428034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 49), 'int')
        # Processing the call keyword arguments (line 29)
        kwargs_428035 = {}
        # Getting the type of 'assert_allclose' (line 29)
        assert_allclose_428025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 29)
        assert_allclose_call_result_428036 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert_allclose_428025, *[spnorm_call_result_428033, int_428034], **kwargs_428035)
        
        
        # Call to assert_allclose(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to spnorm(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'self' (line 30)
        self_428039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'self', False)
        # Obtaining the member 'b' of a type (line 30)
        b_428040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 31), self_428039, 'b')
        int_428041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'int')
        # Processing the call keyword arguments (line 30)
        kwargs_428042 = {}
        # Getting the type of 'spnorm' (line 30)
        spnorm_428038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 30)
        spnorm_call_result_428043 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), spnorm_428038, *[b_428040, int_428041], **kwargs_428042)
        
        int_428044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'int')
        # Processing the call keyword arguments (line 30)
        kwargs_428045 = {}
        # Getting the type of 'assert_allclose' (line 30)
        assert_allclose_428037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 30)
        assert_allclose_call_result_428046 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_allclose_428037, *[spnorm_call_result_428043, int_428044], **kwargs_428045)
        
        
        # Call to assert_allclose(...): (line 31)
        # Processing the call arguments (line 31)
        
        # Call to spnorm(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'self' (line 31)
        self_428049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'self', False)
        # Obtaining the member 'b' of a type (line 31)
        b_428050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 31), self_428049, 'b')
        int_428051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 39), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_428052 = {}
        # Getting the type of 'spnorm' (line 31)
        spnorm_428048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 31)
        spnorm_call_result_428053 = invoke(stypy.reporting.localization.Localization(__file__, 31, 24), spnorm_428048, *[b_428050, int_428051], **kwargs_428052)
        
        int_428054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 44), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_428055 = {}
        # Getting the type of 'assert_allclose' (line 31)
        assert_allclose_428047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 31)
        assert_allclose_call_result_428056 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert_allclose_428047, *[spnorm_call_result_428053, int_428054], **kwargs_428055)
        
        
        # Call to assert_raises(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'NotImplementedError' (line 34)
        NotImplementedError_428058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'NotImplementedError', False)
        # Getting the type of 'spnorm' (line 34)
        spnorm_428059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 43), 'spnorm', False)
        # Getting the type of 'self' (line 34)
        self_428060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 51), 'self', False)
        # Obtaining the member 'b' of a type (line 34)
        b_428061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 51), self_428060, 'b')
        int_428062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 59), 'int')
        # Processing the call keyword arguments (line 34)
        kwargs_428063 = {}
        # Getting the type of 'assert_raises' (line 34)
        assert_raises_428057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 34)
        assert_raises_call_result_428064 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert_raises_428057, *[NotImplementedError_428058, spnorm_428059, b_428061, int_428062], **kwargs_428063)
        
        
        # Call to assert_raises(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'NotImplementedError' (line 35)
        NotImplementedError_428066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'NotImplementedError', False)
        # Getting the type of 'spnorm' (line 35)
        spnorm_428067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 'spnorm', False)
        # Getting the type of 'self' (line 35)
        self_428068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 51), 'self', False)
        # Obtaining the member 'b' of a type (line 35)
        b_428069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 51), self_428068, 'b')
        int_428070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 59), 'int')
        # Processing the call keyword arguments (line 35)
        kwargs_428071 = {}
        # Getting the type of 'assert_raises' (line 35)
        assert_raises_428065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 35)
        assert_raises_call_result_428072 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert_raises_428065, *[NotImplementedError_428066, spnorm_428067, b_428069, int_428070], **kwargs_428071)
        
        
        # ################# End of 'test_matrix_norm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matrix_norm' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_428073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428073)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matrix_norm'
        return stypy_return_type_428073


    @norecursion
    def test_matrix_norm_axis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matrix_norm_axis'
        module_type_store = module_type_store.open_function_context('test_matrix_norm_axis', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_localization', localization)
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_function_name', 'TestNorm.test_matrix_norm_axis')
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_param_names_list', [])
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNorm.test_matrix_norm_axis.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNorm.test_matrix_norm_axis', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matrix_norm_axis', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matrix_norm_axis(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_428074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_428075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        # Getting the type of 'self' (line 38)
        self_428076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'self')
        # Obtaining the member 'b' of a type (line 38)
        b_428077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 25), self_428076, 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 25), tuple_428075, b_428077)
        # Adding element type (line 38)
        # Getting the type of 'None' (line 38)
        None_428078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 25), tuple_428075, None_428078)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_428074, tuple_428075)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_428079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        # Getting the type of 'self' (line 38)
        self_428080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 41), 'self')
        # Obtaining the member 'b' of a type (line 38)
        b_428081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 41), self_428080, 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 41), tuple_428079, b_428081)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_428082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        int_428083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 50), tuple_428082, int_428083)
        # Adding element type (line 38)
        int_428084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 50), tuple_428082, int_428084)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 41), tuple_428079, tuple_428082)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_428074, tuple_428079)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_428085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        # Getting the type of 'self' (line 38)
        self_428086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 59), 'self')
        # Obtaining the member 'b' of a type (line 38)
        b_428087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 59), self_428086, 'b')
        # Obtaining the member 'T' of a type (line 38)
        T_428088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 59), b_428087, 'T')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 59), tuple_428085, T_428088)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_428089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 70), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        int_428090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 70), tuple_428089, int_428090)
        # Adding element type (line 38)
        int_428091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 73), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 70), tuple_428089, int_428091)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 59), tuple_428085, tuple_428089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_428074, tuple_428085)
        
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), tuple_428074)
        # Getting the type of the for loop variable (line 38)
        for_loop_var_428092 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), tuple_428074)
        # Assigning a type to the variable 'm' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 8), for_loop_var_428092))
        # Assigning a type to the variable 'axis' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 8), for_loop_var_428092))
        # SSA begins for a for statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to spnorm(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'm' (line 39)
        m_428095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 35), 'm', False)
        # Processing the call keyword arguments (line 39)
        # Getting the type of 'axis' (line 39)
        axis_428096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 43), 'axis', False)
        keyword_428097 = axis_428096
        kwargs_428098 = {'axis': keyword_428097}
        # Getting the type of 'spnorm' (line 39)
        spnorm_428094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 39)
        spnorm_call_result_428099 = invoke(stypy.reporting.localization.Localization(__file__, 39, 28), spnorm_428094, *[m_428095], **kwargs_428098)
        
        float_428100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'float')
        # Processing the call keyword arguments (line 39)
        kwargs_428101 = {}
        # Getting the type of 'assert_allclose' (line 39)
        assert_allclose_428093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 39)
        assert_allclose_call_result_428102 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), assert_allclose_428093, *[spnorm_call_result_428099, float_428100], **kwargs_428101)
        
        
        # Call to assert_allclose(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to spnorm(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'm' (line 40)
        m_428105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'm', False)
        str_428106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'str', 'fro')
        # Processing the call keyword arguments (line 40)
        # Getting the type of 'axis' (line 40)
        axis_428107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 50), 'axis', False)
        keyword_428108 = axis_428107
        kwargs_428109 = {'axis': keyword_428108}
        # Getting the type of 'spnorm' (line 40)
        spnorm_428104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 40)
        spnorm_call_result_428110 = invoke(stypy.reporting.localization.Localization(__file__, 40, 28), spnorm_428104, *[m_428105, str_428106], **kwargs_428109)
        
        float_428111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 57), 'float')
        # Processing the call keyword arguments (line 40)
        kwargs_428112 = {}
        # Getting the type of 'assert_allclose' (line 40)
        assert_allclose_428103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 40)
        assert_allclose_call_result_428113 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), assert_allclose_428103, *[spnorm_call_result_428110, float_428111], **kwargs_428112)
        
        
        # Call to assert_allclose(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to spnorm(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'm' (line 41)
        m_428116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'm', False)
        # Getting the type of 'np' (line 41)
        np_428117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'np', False)
        # Obtaining the member 'inf' of a type (line 41)
        inf_428118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 38), np_428117, 'inf')
        # Processing the call keyword arguments (line 41)
        # Getting the type of 'axis' (line 41)
        axis_428119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 51), 'axis', False)
        keyword_428120 = axis_428119
        kwargs_428121 = {'axis': keyword_428120}
        # Getting the type of 'spnorm' (line 41)
        spnorm_428115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 41)
        spnorm_call_result_428122 = invoke(stypy.reporting.localization.Localization(__file__, 41, 28), spnorm_428115, *[m_428116, inf_428118], **kwargs_428121)
        
        int_428123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 58), 'int')
        # Processing the call keyword arguments (line 41)
        kwargs_428124 = {}
        # Getting the type of 'assert_allclose' (line 41)
        assert_allclose_428114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 41)
        assert_allclose_call_result_428125 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), assert_allclose_428114, *[spnorm_call_result_428122, int_428123], **kwargs_428124)
        
        
        # Call to assert_allclose(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to spnorm(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'm' (line 42)
        m_428128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 35), 'm', False)
        
        # Getting the type of 'np' (line 42)
        np_428129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'np', False)
        # Obtaining the member 'inf' of a type (line 42)
        inf_428130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), np_428129, 'inf')
        # Applying the 'usub' unary operator (line 42)
        result___neg___428131 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 38), 'usub', inf_428130)
        
        # Processing the call keyword arguments (line 42)
        # Getting the type of 'axis' (line 42)
        axis_428132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 52), 'axis', False)
        keyword_428133 = axis_428132
        kwargs_428134 = {'axis': keyword_428133}
        # Getting the type of 'spnorm' (line 42)
        spnorm_428127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 42)
        spnorm_call_result_428135 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), spnorm_428127, *[m_428128, result___neg___428131], **kwargs_428134)
        
        int_428136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 59), 'int')
        # Processing the call keyword arguments (line 42)
        kwargs_428137 = {}
        # Getting the type of 'assert_allclose' (line 42)
        assert_allclose_428126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 42)
        assert_allclose_call_result_428138 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), assert_allclose_428126, *[spnorm_call_result_428135, int_428136], **kwargs_428137)
        
        
        # Call to assert_allclose(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to spnorm(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'm' (line 43)
        m_428141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 35), 'm', False)
        int_428142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 38), 'int')
        # Processing the call keyword arguments (line 43)
        # Getting the type of 'axis' (line 43)
        axis_428143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 46), 'axis', False)
        keyword_428144 = axis_428143
        kwargs_428145 = {'axis': keyword_428144}
        # Getting the type of 'spnorm' (line 43)
        spnorm_428140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 43)
        spnorm_call_result_428146 = invoke(stypy.reporting.localization.Localization(__file__, 43, 28), spnorm_428140, *[m_428141, int_428142], **kwargs_428145)
        
        int_428147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 53), 'int')
        # Processing the call keyword arguments (line 43)
        kwargs_428148 = {}
        # Getting the type of 'assert_allclose' (line 43)
        assert_allclose_428139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 43)
        assert_allclose_call_result_428149 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), assert_allclose_428139, *[spnorm_call_result_428146, int_428147], **kwargs_428148)
        
        
        # Call to assert_allclose(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Call to spnorm(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'm' (line 44)
        m_428152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'm', False)
        int_428153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'int')
        # Processing the call keyword arguments (line 44)
        # Getting the type of 'axis' (line 44)
        axis_428154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 47), 'axis', False)
        keyword_428155 = axis_428154
        kwargs_428156 = {'axis': keyword_428155}
        # Getting the type of 'spnorm' (line 44)
        spnorm_428151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 28), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 44)
        spnorm_call_result_428157 = invoke(stypy.reporting.localization.Localization(__file__, 44, 28), spnorm_428151, *[m_428152, int_428153], **kwargs_428156)
        
        int_428158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 54), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_428159 = {}
        # Getting the type of 'assert_allclose' (line 44)
        assert_allclose_428150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 44)
        assert_allclose_call_result_428160 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), assert_allclose_428150, *[spnorm_call_result_428157, int_428158], **kwargs_428159)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_matrix_norm_axis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matrix_norm_axis' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_428161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428161)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matrix_norm_axis'
        return stypy_return_type_428161


    @norecursion
    def test_vector_norm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vector_norm'
        module_type_store = module_type_store.open_function_context('test_vector_norm', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_localization', localization)
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_function_name', 'TestNorm.test_vector_norm')
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_param_names_list', [])
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNorm.test_vector_norm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNorm.test_vector_norm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vector_norm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vector_norm(...)' code ##################

        
        # Assigning a List to a Name (line 47):
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_428162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        float_428163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 12), list_428162, float_428163)
        # Adding element type (line 47)
        float_428164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 12), list_428162, float_428164)
        # Adding element type (line 47)
        float_428165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 12), list_428162, float_428165)
        
        # Assigning a type to the variable 'v' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'v', list_428162)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 48)
        tuple_428166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 48)
        # Adding element type (line 48)
        
        # Obtaining an instance of the builtin type 'tuple' (line 48)
        tuple_428167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 48)
        # Adding element type (line 48)
        # Getting the type of 'self' (line 48)
        self_428168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'self')
        # Obtaining the member 'b' of a type (line 48)
        b_428169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 21), self_428168, 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 21), tuple_428167, b_428169)
        # Adding element type (line 48)
        int_428170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 21), tuple_428167, int_428170)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), tuple_428166, tuple_428167)
        # Adding element type (line 48)
        
        # Obtaining an instance of the builtin type 'tuple' (line 48)
        tuple_428171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 48)
        # Adding element type (line 48)
        # Getting the type of 'self' (line 48)
        self_428172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'self')
        # Obtaining the member 'b' of a type (line 48)
        b_428173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 34), self_428172, 'b')
        # Obtaining the member 'T' of a type (line 48)
        T_428174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 34), b_428173, 'T')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 34), tuple_428171, T_428174)
        # Adding element type (line 48)
        int_428175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 34), tuple_428171, int_428175)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), tuple_428166, tuple_428171)
        
        # Testing the type of a for loop iterable (line 48)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 8), tuple_428166)
        # Getting the type of the for loop variable (line 48)
        for_loop_var_428176 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 8), tuple_428166)
        # Assigning a type to the variable 'm' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 8), for_loop_var_428176))
        # Assigning a type to the variable 'a' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 8), for_loop_var_428176))
        # SSA begins for a for statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_428177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'a' (line 49)
        a_428178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), tuple_428177, a_428178)
        # Adding element type (line 49)
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_428179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'a' (line 49)
        a_428180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 28), tuple_428179, a_428180)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), tuple_428177, tuple_428179)
        # Adding element type (line 49)
        # Getting the type of 'a' (line 49)
        a_428181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'a')
        int_428182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_428183 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 34), '-', a_428181, int_428182)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), tuple_428177, result_sub_428183)
        # Adding element type (line 49)
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_428184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'a' (line 49)
        a_428185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'a')
        int_428186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 42), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_428187 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 40), '-', a_428185, int_428186)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 40), tuple_428184, result_sub_428187)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), tuple_428177, tuple_428184)
        
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 12), tuple_428177)
        # Getting the type of the for loop variable (line 49)
        for_loop_var_428188 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 12), tuple_428177)
        # Assigning a type to the variable 'axis' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'axis', for_loop_var_428188)
        # SSA begins for a for statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to spnorm(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'm' (line 50)
        m_428191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 39), 'm', False)
        int_428192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 42), 'int')
        # Processing the call keyword arguments (line 50)
        # Getting the type of 'axis' (line 50)
        axis_428193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 50), 'axis', False)
        keyword_428194 = axis_428193
        kwargs_428195 = {'axis': keyword_428194}
        # Getting the type of 'spnorm' (line 50)
        spnorm_428190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 50)
        spnorm_call_result_428196 = invoke(stypy.reporting.localization.Localization(__file__, 50, 32), spnorm_428190, *[m_428191, int_428192], **kwargs_428195)
        
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_428197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        int_428198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 57), list_428197, int_428198)
        # Adding element type (line 50)
        int_428199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 57), list_428197, int_428199)
        # Adding element type (line 50)
        int_428200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 57), list_428197, int_428200)
        
        # Processing the call keyword arguments (line 50)
        kwargs_428201 = {}
        # Getting the type of 'assert_allclose' (line 50)
        assert_allclose_428189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 50)
        assert_allclose_call_result_428202 = invoke(stypy.reporting.localization.Localization(__file__, 50, 16), assert_allclose_428189, *[spnorm_call_result_428196, list_428197], **kwargs_428201)
        
        
        # Call to assert_allclose(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to spnorm(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'm' (line 51)
        m_428205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 39), 'm', False)
        # Getting the type of 'np' (line 51)
        np_428206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'np', False)
        # Obtaining the member 'inf' of a type (line 51)
        inf_428207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 42), np_428206, 'inf')
        # Processing the call keyword arguments (line 51)
        # Getting the type of 'axis' (line 51)
        axis_428208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 55), 'axis', False)
        keyword_428209 = axis_428208
        kwargs_428210 = {'axis': keyword_428209}
        # Getting the type of 'spnorm' (line 51)
        spnorm_428204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 51)
        spnorm_call_result_428211 = invoke(stypy.reporting.localization.Localization(__file__, 51, 32), spnorm_428204, *[m_428205, inf_428207], **kwargs_428210)
        
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_428212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        int_428213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 62), list_428212, int_428213)
        # Adding element type (line 51)
        int_428214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 62), list_428212, int_428214)
        # Adding element type (line 51)
        int_428215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 62), list_428212, int_428215)
        
        # Processing the call keyword arguments (line 51)
        kwargs_428216 = {}
        # Getting the type of 'assert_allclose' (line 51)
        assert_allclose_428203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 51)
        assert_allclose_call_result_428217 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), assert_allclose_428203, *[spnorm_call_result_428211, list_428212], **kwargs_428216)
        
        
        # Call to assert_allclose(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Call to spnorm(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'm' (line 52)
        m_428220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'm', False)
        # Processing the call keyword arguments (line 52)
        # Getting the type of 'axis' (line 52)
        axis_428221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 47), 'axis', False)
        keyword_428222 = axis_428221
        kwargs_428223 = {'axis': keyword_428222}
        # Getting the type of 'spnorm' (line 52)
        spnorm_428219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 52)
        spnorm_call_result_428224 = invoke(stypy.reporting.localization.Localization(__file__, 52, 32), spnorm_428219, *[m_428220], **kwargs_428223)
        
        # Getting the type of 'v' (line 52)
        v_428225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 54), 'v', False)
        # Processing the call keyword arguments (line 52)
        kwargs_428226 = {}
        # Getting the type of 'assert_allclose' (line 52)
        assert_allclose_428218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 52)
        assert_allclose_call_result_428227 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), assert_allclose_428218, *[spnorm_call_result_428224, v_428225], **kwargs_428226)
        
        
        # Call to assert_allclose(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to spnorm(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'm' (line 53)
        m_428230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), 'm', False)
        # Processing the call keyword arguments (line 53)
        int_428231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 46), 'int')
        keyword_428232 = int_428231
        # Getting the type of 'axis' (line 53)
        axis_428233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 54), 'axis', False)
        keyword_428234 = axis_428233
        kwargs_428235 = {'ord': keyword_428232, 'axis': keyword_428234}
        # Getting the type of 'spnorm' (line 53)
        spnorm_428229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 53)
        spnorm_call_result_428236 = invoke(stypy.reporting.localization.Localization(__file__, 53, 32), spnorm_428229, *[m_428230], **kwargs_428235)
        
        # Getting the type of 'v' (line 53)
        v_428237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 61), 'v', False)
        # Processing the call keyword arguments (line 53)
        kwargs_428238 = {}
        # Getting the type of 'assert_allclose' (line 53)
        assert_allclose_428228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 53)
        assert_allclose_call_result_428239 = invoke(stypy.reporting.localization.Localization(__file__, 53, 16), assert_allclose_428228, *[spnorm_call_result_428236, v_428237], **kwargs_428238)
        
        
        # Call to assert_allclose(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to spnorm(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'm' (line 54)
        m_428242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'm', False)
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'None' (line 54)
        None_428243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 46), 'None', False)
        keyword_428244 = None_428243
        # Getting the type of 'axis' (line 54)
        axis_428245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 57), 'axis', False)
        keyword_428246 = axis_428245
        kwargs_428247 = {'ord': keyword_428244, 'axis': keyword_428246}
        # Getting the type of 'spnorm' (line 54)
        spnorm_428241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 54)
        spnorm_call_result_428248 = invoke(stypy.reporting.localization.Localization(__file__, 54, 32), spnorm_428241, *[m_428242], **kwargs_428247)
        
        # Getting the type of 'v' (line 54)
        v_428249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 64), 'v', False)
        # Processing the call keyword arguments (line 54)
        kwargs_428250 = {}
        # Getting the type of 'assert_allclose' (line 54)
        assert_allclose_428240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 54)
        assert_allclose_call_result_428251 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), assert_allclose_428240, *[spnorm_call_result_428248, v_428249], **kwargs_428250)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_vector_norm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vector_norm' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_428252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vector_norm'
        return stypy_return_type_428252


    @norecursion
    def test_norm_exceptions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_norm_exceptions'
        module_type_store = module_type_store.open_function_context('test_norm_exceptions', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_localization', localization)
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_function_name', 'TestNorm.test_norm_exceptions')
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_param_names_list', [])
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNorm.test_norm_exceptions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNorm.test_norm_exceptions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_norm_exceptions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_norm_exceptions(...)' code ##################

        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'self' (line 57)
        self_428253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'self')
        # Obtaining the member 'b' of a type (line 57)
        b_428254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), self_428253, 'b')
        # Assigning a type to the variable 'm' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'm', b_428254)
        
        # Call to assert_raises(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'TypeError' (line 58)
        TypeError_428256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'TypeError', False)
        # Getting the type of 'spnorm' (line 58)
        spnorm_428257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 33), 'spnorm', False)
        # Getting the type of 'm' (line 58)
        m_428258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 41), 'm', False)
        # Getting the type of 'None' (line 58)
        None_428259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 44), 'None', False)
        float_428260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 50), 'float')
        # Processing the call keyword arguments (line 58)
        kwargs_428261 = {}
        # Getting the type of 'assert_raises' (line 58)
        assert_raises_428255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 58)
        assert_raises_call_result_428262 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_raises_428255, *[TypeError_428256, spnorm_428257, m_428258, None_428259, float_428260], **kwargs_428261)
        
        
        # Call to assert_raises(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'TypeError' (line 59)
        TypeError_428264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'TypeError', False)
        # Getting the type of 'spnorm' (line 59)
        spnorm_428265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'spnorm', False)
        # Getting the type of 'm' (line 59)
        m_428266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 41), 'm', False)
        # Getting the type of 'None' (line 59)
        None_428267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), 'None', False)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_428268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        int_428269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 50), list_428268, int_428269)
        
        # Processing the call keyword arguments (line 59)
        kwargs_428270 = {}
        # Getting the type of 'assert_raises' (line 59)
        assert_raises_428263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 59)
        assert_raises_call_result_428271 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert_raises_428263, *[TypeError_428264, spnorm_428265, m_428266, None_428267, list_428268], **kwargs_428270)
        
        
        # Call to assert_raises(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'ValueError' (line 60)
        ValueError_428273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 60)
        spnorm_428274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'spnorm', False)
        # Getting the type of 'm' (line 60)
        m_428275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 42), 'm', False)
        # Getting the type of 'None' (line 60)
        None_428276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'None', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 60)
        tuple_428277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 60)
        
        # Processing the call keyword arguments (line 60)
        kwargs_428278 = {}
        # Getting the type of 'assert_raises' (line 60)
        assert_raises_428272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 60)
        assert_raises_call_result_428279 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_raises_428272, *[ValueError_428273, spnorm_428274, m_428275, None_428276, tuple_428277], **kwargs_428278)
        
        
        # Call to assert_raises(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'ValueError' (line 61)
        ValueError_428281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 61)
        spnorm_428282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'spnorm', False)
        # Getting the type of 'm' (line 61)
        m_428283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'm', False)
        # Getting the type of 'None' (line 61)
        None_428284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'None', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 61)
        tuple_428285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 61)
        # Adding element type (line 61)
        int_428286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 52), tuple_428285, int_428286)
        # Adding element type (line 61)
        int_428287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 52), tuple_428285, int_428287)
        # Adding element type (line 61)
        int_428288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 52), tuple_428285, int_428288)
        
        # Processing the call keyword arguments (line 61)
        kwargs_428289 = {}
        # Getting the type of 'assert_raises' (line 61)
        assert_raises_428280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 61)
        assert_raises_call_result_428290 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_raises_428280, *[ValueError_428281, spnorm_428282, m_428283, None_428284, tuple_428285], **kwargs_428289)
        
        
        # Call to assert_raises(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'ValueError' (line 62)
        ValueError_428292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 62)
        spnorm_428293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 34), 'spnorm', False)
        # Getting the type of 'm' (line 62)
        m_428294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 42), 'm', False)
        # Getting the type of 'None' (line 62)
        None_428295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 45), 'None', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_428296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        int_428297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 52), tuple_428296, int_428297)
        # Adding element type (line 62)
        int_428298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 52), tuple_428296, int_428298)
        
        # Processing the call keyword arguments (line 62)
        kwargs_428299 = {}
        # Getting the type of 'assert_raises' (line 62)
        assert_raises_428291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 62)
        assert_raises_call_result_428300 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_raises_428291, *[ValueError_428292, spnorm_428293, m_428294, None_428295, tuple_428296], **kwargs_428299)
        
        
        # Call to assert_raises(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'ValueError' (line 63)
        ValueError_428302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 63)
        spnorm_428303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), 'spnorm', False)
        # Getting the type of 'm' (line 63)
        m_428304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 42), 'm', False)
        # Getting the type of 'None' (line 63)
        None_428305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 45), 'None', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_428306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        int_428307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 52), tuple_428306, int_428307)
        # Adding element type (line 63)
        int_428308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 52), tuple_428306, int_428308)
        
        # Processing the call keyword arguments (line 63)
        kwargs_428309 = {}
        # Getting the type of 'assert_raises' (line 63)
        assert_raises_428301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 63)
        assert_raises_call_result_428310 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), assert_raises_428301, *[ValueError_428302, spnorm_428303, m_428304, None_428305, tuple_428306], **kwargs_428309)
        
        
        # Call to assert_raises(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'ValueError' (line 64)
        ValueError_428312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 64)
        spnorm_428313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'spnorm', False)
        # Getting the type of 'm' (line 64)
        m_428314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 42), 'm', False)
        # Getting the type of 'None' (line 64)
        None_428315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 45), 'None', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 64)
        tuple_428316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 64)
        # Adding element type (line 64)
        int_428317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 52), tuple_428316, int_428317)
        # Adding element type (line 64)
        int_428318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 52), tuple_428316, int_428318)
        
        # Processing the call keyword arguments (line 64)
        kwargs_428319 = {}
        # Getting the type of 'assert_raises' (line 64)
        assert_raises_428311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 64)
        assert_raises_call_result_428320 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_raises_428311, *[ValueError_428312, spnorm_428313, m_428314, None_428315, tuple_428316], **kwargs_428319)
        
        
        # Call to assert_raises(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'ValueError' (line 65)
        ValueError_428322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 65)
        spnorm_428323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'spnorm', False)
        # Getting the type of 'm' (line 65)
        m_428324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'm', False)
        # Getting the type of 'None' (line 65)
        None_428325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 45), 'None', False)
        int_428326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 51), 'int')
        # Processing the call keyword arguments (line 65)
        kwargs_428327 = {}
        # Getting the type of 'assert_raises' (line 65)
        assert_raises_428321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 65)
        assert_raises_call_result_428328 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_raises_428321, *[ValueError_428322, spnorm_428323, m_428324, None_428325, int_428326], **kwargs_428327)
        
        
        # Call to assert_raises(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'ValueError' (line 66)
        ValueError_428330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 66)
        spnorm_428331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'spnorm', False)
        # Getting the type of 'm' (line 66)
        m_428332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'm', False)
        # Getting the type of 'None' (line 66)
        None_428333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'None', False)
        int_428334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 51), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_428335 = {}
        # Getting the type of 'assert_raises' (line 66)
        assert_raises_428329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 66)
        assert_raises_call_result_428336 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_raises_428329, *[ValueError_428330, spnorm_428331, m_428332, None_428333, int_428334], **kwargs_428335)
        
        
        # Call to assert_raises(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'ValueError' (line 67)
        ValueError_428338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 67)
        spnorm_428339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'spnorm', False)
        # Getting the type of 'm' (line 67)
        m_428340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 42), 'm', False)
        str_428341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 45), 'str', 'plate_of_shrimp')
        int_428342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 64), 'int')
        # Processing the call keyword arguments (line 67)
        kwargs_428343 = {}
        # Getting the type of 'assert_raises' (line 67)
        assert_raises_428337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 67)
        assert_raises_call_result_428344 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assert_raises_428337, *[ValueError_428338, spnorm_428339, m_428340, str_428341, int_428342], **kwargs_428343)
        
        
        # Call to assert_raises(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'ValueError' (line 68)
        ValueError_428346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'ValueError', False)
        # Getting the type of 'spnorm' (line 68)
        spnorm_428347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'spnorm', False)
        # Getting the type of 'm' (line 68)
        m_428348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 42), 'm', False)
        str_428349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 45), 'str', 'plate_of_shrimp')
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_428350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 65), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        int_428351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 65), tuple_428350, int_428351)
        # Adding element type (line 68)
        int_428352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 65), tuple_428350, int_428352)
        
        # Processing the call keyword arguments (line 68)
        kwargs_428353 = {}
        # Getting the type of 'assert_raises' (line 68)
        assert_raises_428345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 68)
        assert_raises_call_result_428354 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assert_raises_428345, *[ValueError_428346, spnorm_428347, m_428348, str_428349, tuple_428350], **kwargs_428353)
        
        
        # ################# End of 'test_norm_exceptions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_norm_exceptions' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_428355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428355)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_norm_exceptions'
        return stypy_return_type_428355


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 0, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNorm.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestNorm' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'TestNorm', TestNorm)
# Declaration of the 'TestVsNumpyNorm' class

class TestVsNumpyNorm(object, ):

    @norecursion
    def test_sparse_matrix_norms(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_matrix_norms'
        module_type_store = module_type_store.open_function_context('test_sparse_matrix_norms', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_localization', localization)
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_function_name', 'TestVsNumpyNorm.test_sparse_matrix_norms')
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_param_names_list', [])
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVsNumpyNorm.test_sparse_matrix_norms.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVsNumpyNorm.test_sparse_matrix_norms', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_matrix_norms', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_matrix_norms(...)' code ##################

        
        # Getting the type of 'self' (line 92)
        self_428356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'self')
        # Obtaining the member '_sparse_types' of a type (line 92)
        _sparse_types_428357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 27), self_428356, '_sparse_types')
        # Testing the type of a for loop iterable (line 92)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 92, 8), _sparse_types_428357)
        # Getting the type of the for loop variable (line 92)
        for_loop_var_428358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 92, 8), _sparse_types_428357)
        # Assigning a type to the variable 'sparse_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'sparse_type', for_loop_var_428358)
        # SSA begins for a for statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 93)
        self_428359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'self')
        # Obtaining the member '_test_matrices' of a type (line 93)
        _test_matrices_428360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 21), self_428359, '_test_matrices')
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 12), _test_matrices_428360)
        # Getting the type of the for loop variable (line 93)
        for_loop_var_428361 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 12), _test_matrices_428360)
        # Assigning a type to the variable 'M' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'M', for_loop_var_428361)
        # SSA begins for a for statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 94):
        
        # Call to sparse_type(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'M' (line 94)
        M_428363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'M', False)
        # Processing the call keyword arguments (line 94)
        kwargs_428364 = {}
        # Getting the type of 'sparse_type' (line 94)
        sparse_type_428362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'sparse_type', False)
        # Calling sparse_type(args, kwargs) (line 94)
        sparse_type_call_result_428365 = invoke(stypy.reporting.localization.Localization(__file__, 94, 20), sparse_type_428362, *[M_428363], **kwargs_428364)
        
        # Assigning a type to the variable 'S' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'S', sparse_type_call_result_428365)
        
        # Call to assert_allclose(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Call to spnorm(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'S' (line 95)
        S_428368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 39), 'S', False)
        # Processing the call keyword arguments (line 95)
        kwargs_428369 = {}
        # Getting the type of 'spnorm' (line 95)
        spnorm_428367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 95)
        spnorm_call_result_428370 = invoke(stypy.reporting.localization.Localization(__file__, 95, 32), spnorm_428367, *[S_428368], **kwargs_428369)
        
        
        # Call to npnorm(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'M' (line 95)
        M_428372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 50), 'M', False)
        # Processing the call keyword arguments (line 95)
        kwargs_428373 = {}
        # Getting the type of 'npnorm' (line 95)
        npnorm_428371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 43), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 95)
        npnorm_call_result_428374 = invoke(stypy.reporting.localization.Localization(__file__, 95, 43), npnorm_428371, *[M_428372], **kwargs_428373)
        
        # Processing the call keyword arguments (line 95)
        kwargs_428375 = {}
        # Getting the type of 'assert_allclose' (line 95)
        assert_allclose_428366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 95)
        assert_allclose_call_result_428376 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), assert_allclose_428366, *[spnorm_call_result_428370, npnorm_call_result_428374], **kwargs_428375)
        
        
        # Call to assert_allclose(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to spnorm(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'S' (line 96)
        S_428379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 39), 'S', False)
        str_428380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'str', 'fro')
        # Processing the call keyword arguments (line 96)
        kwargs_428381 = {}
        # Getting the type of 'spnorm' (line 96)
        spnorm_428378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 96)
        spnorm_call_result_428382 = invoke(stypy.reporting.localization.Localization(__file__, 96, 32), spnorm_428378, *[S_428379, str_428380], **kwargs_428381)
        
        
        # Call to npnorm(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'M' (line 96)
        M_428384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 57), 'M', False)
        str_428385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 60), 'str', 'fro')
        # Processing the call keyword arguments (line 96)
        kwargs_428386 = {}
        # Getting the type of 'npnorm' (line 96)
        npnorm_428383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 50), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 96)
        npnorm_call_result_428387 = invoke(stypy.reporting.localization.Localization(__file__, 96, 50), npnorm_428383, *[M_428384, str_428385], **kwargs_428386)
        
        # Processing the call keyword arguments (line 96)
        kwargs_428388 = {}
        # Getting the type of 'assert_allclose' (line 96)
        assert_allclose_428377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 96)
        assert_allclose_call_result_428389 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), assert_allclose_428377, *[spnorm_call_result_428382, npnorm_call_result_428387], **kwargs_428388)
        
        
        # Call to assert_allclose(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to spnorm(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'S' (line 97)
        S_428392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'S', False)
        # Getting the type of 'np' (line 97)
        np_428393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'np', False)
        # Obtaining the member 'inf' of a type (line 97)
        inf_428394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 42), np_428393, 'inf')
        # Processing the call keyword arguments (line 97)
        kwargs_428395 = {}
        # Getting the type of 'spnorm' (line 97)
        spnorm_428391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 97)
        spnorm_call_result_428396 = invoke(stypy.reporting.localization.Localization(__file__, 97, 32), spnorm_428391, *[S_428392, inf_428394], **kwargs_428395)
        
        
        # Call to npnorm(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'M' (line 97)
        M_428398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 58), 'M', False)
        # Getting the type of 'np' (line 97)
        np_428399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 61), 'np', False)
        # Obtaining the member 'inf' of a type (line 97)
        inf_428400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 61), np_428399, 'inf')
        # Processing the call keyword arguments (line 97)
        kwargs_428401 = {}
        # Getting the type of 'npnorm' (line 97)
        npnorm_428397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 51), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 97)
        npnorm_call_result_428402 = invoke(stypy.reporting.localization.Localization(__file__, 97, 51), npnorm_428397, *[M_428398, inf_428400], **kwargs_428401)
        
        # Processing the call keyword arguments (line 97)
        kwargs_428403 = {}
        # Getting the type of 'assert_allclose' (line 97)
        assert_allclose_428390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 97)
        assert_allclose_call_result_428404 = invoke(stypy.reporting.localization.Localization(__file__, 97, 16), assert_allclose_428390, *[spnorm_call_result_428396, npnorm_call_result_428402], **kwargs_428403)
        
        
        # Call to assert_allclose(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to spnorm(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'S' (line 98)
        S_428407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 39), 'S', False)
        
        # Getting the type of 'np' (line 98)
        np_428408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 43), 'np', False)
        # Obtaining the member 'inf' of a type (line 98)
        inf_428409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 43), np_428408, 'inf')
        # Applying the 'usub' unary operator (line 98)
        result___neg___428410 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 42), 'usub', inf_428409)
        
        # Processing the call keyword arguments (line 98)
        kwargs_428411 = {}
        # Getting the type of 'spnorm' (line 98)
        spnorm_428406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 98)
        spnorm_call_result_428412 = invoke(stypy.reporting.localization.Localization(__file__, 98, 32), spnorm_428406, *[S_428407, result___neg___428410], **kwargs_428411)
        
        
        # Call to npnorm(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'M' (line 98)
        M_428414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 59), 'M', False)
        
        # Getting the type of 'np' (line 98)
        np_428415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 63), 'np', False)
        # Obtaining the member 'inf' of a type (line 98)
        inf_428416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 63), np_428415, 'inf')
        # Applying the 'usub' unary operator (line 98)
        result___neg___428417 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 62), 'usub', inf_428416)
        
        # Processing the call keyword arguments (line 98)
        kwargs_428418 = {}
        # Getting the type of 'npnorm' (line 98)
        npnorm_428413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 52), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 98)
        npnorm_call_result_428419 = invoke(stypy.reporting.localization.Localization(__file__, 98, 52), npnorm_428413, *[M_428414, result___neg___428417], **kwargs_428418)
        
        # Processing the call keyword arguments (line 98)
        kwargs_428420 = {}
        # Getting the type of 'assert_allclose' (line 98)
        assert_allclose_428405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 98)
        assert_allclose_call_result_428421 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), assert_allclose_428405, *[spnorm_call_result_428412, npnorm_call_result_428419], **kwargs_428420)
        
        
        # Call to assert_allclose(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to spnorm(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'S' (line 99)
        S_428424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 39), 'S', False)
        int_428425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 42), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_428426 = {}
        # Getting the type of 'spnorm' (line 99)
        spnorm_428423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 99)
        spnorm_call_result_428427 = invoke(stypy.reporting.localization.Localization(__file__, 99, 32), spnorm_428423, *[S_428424, int_428425], **kwargs_428426)
        
        
        # Call to npnorm(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'M' (line 99)
        M_428429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 53), 'M', False)
        int_428430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 56), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_428431 = {}
        # Getting the type of 'npnorm' (line 99)
        npnorm_428428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 46), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 99)
        npnorm_call_result_428432 = invoke(stypy.reporting.localization.Localization(__file__, 99, 46), npnorm_428428, *[M_428429, int_428430], **kwargs_428431)
        
        # Processing the call keyword arguments (line 99)
        kwargs_428433 = {}
        # Getting the type of 'assert_allclose' (line 99)
        assert_allclose_428422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 99)
        assert_allclose_call_result_428434 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), assert_allclose_428422, *[spnorm_call_result_428427, npnorm_call_result_428432], **kwargs_428433)
        
        
        # Call to assert_allclose(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to spnorm(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'S' (line 100)
        S_428437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'S', False)
        int_428438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 42), 'int')
        # Processing the call keyword arguments (line 100)
        kwargs_428439 = {}
        # Getting the type of 'spnorm' (line 100)
        spnorm_428436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 100)
        spnorm_call_result_428440 = invoke(stypy.reporting.localization.Localization(__file__, 100, 32), spnorm_428436, *[S_428437, int_428438], **kwargs_428439)
        
        
        # Call to npnorm(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'M' (line 100)
        M_428442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 54), 'M', False)
        int_428443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 57), 'int')
        # Processing the call keyword arguments (line 100)
        kwargs_428444 = {}
        # Getting the type of 'npnorm' (line 100)
        npnorm_428441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 47), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 100)
        npnorm_call_result_428445 = invoke(stypy.reporting.localization.Localization(__file__, 100, 47), npnorm_428441, *[M_428442, int_428443], **kwargs_428444)
        
        # Processing the call keyword arguments (line 100)
        kwargs_428446 = {}
        # Getting the type of 'assert_allclose' (line 100)
        assert_allclose_428435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 100)
        assert_allclose_call_result_428447 = invoke(stypy.reporting.localization.Localization(__file__, 100, 16), assert_allclose_428435, *[spnorm_call_result_428440, npnorm_call_result_428445], **kwargs_428446)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sparse_matrix_norms(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_matrix_norms' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_428448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428448)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_matrix_norms'
        return stypy_return_type_428448


    @norecursion
    def test_sparse_matrix_norms_with_axis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_matrix_norms_with_axis'
        module_type_store = module_type_store.open_function_context('test_sparse_matrix_norms_with_axis', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_localization', localization)
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_function_name', 'TestVsNumpyNorm.test_sparse_matrix_norms_with_axis')
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_param_names_list', [])
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVsNumpyNorm.test_sparse_matrix_norms_with_axis.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVsNumpyNorm.test_sparse_matrix_norms_with_axis', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_matrix_norms_with_axis', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_matrix_norms_with_axis(...)' code ##################

        
        # Getting the type of 'self' (line 103)
        self_428449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'self')
        # Obtaining the member '_sparse_types' of a type (line 103)
        _sparse_types_428450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), self_428449, '_sparse_types')
        # Testing the type of a for loop iterable (line 103)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 103, 8), _sparse_types_428450)
        # Getting the type of the for loop variable (line 103)
        for_loop_var_428451 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 103, 8), _sparse_types_428450)
        # Assigning a type to the variable 'sparse_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'sparse_type', for_loop_var_428451)
        # SSA begins for a for statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 104)
        self_428452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'self')
        # Obtaining the member '_test_matrices' of a type (line 104)
        _test_matrices_428453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 21), self_428452, '_test_matrices')
        # Testing the type of a for loop iterable (line 104)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 12), _test_matrices_428453)
        # Getting the type of the for loop variable (line 104)
        for_loop_var_428454 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 12), _test_matrices_428453)
        # Assigning a type to the variable 'M' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'M', for_loop_var_428454)
        # SSA begins for a for statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 105):
        
        # Call to sparse_type(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'M' (line 105)
        M_428456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 32), 'M', False)
        # Processing the call keyword arguments (line 105)
        kwargs_428457 = {}
        # Getting the type of 'sparse_type' (line 105)
        sparse_type_428455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'sparse_type', False)
        # Calling sparse_type(args, kwargs) (line 105)
        sparse_type_call_result_428458 = invoke(stypy.reporting.localization.Localization(__file__, 105, 20), sparse_type_428455, *[M_428456], **kwargs_428457)
        
        # Assigning a type to the variable 'S' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'S', sparse_type_call_result_428458)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_428459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        # Getting the type of 'None' (line 106)
        None_428460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), tuple_428459, None_428460)
        # Adding element type (line 106)
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_428461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        int_428462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 35), tuple_428461, int_428462)
        # Adding element type (line 106)
        int_428463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 35), tuple_428461, int_428463)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), tuple_428459, tuple_428461)
        # Adding element type (line 106)
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_428464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        int_428465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 43), tuple_428464, int_428465)
        # Adding element type (line 106)
        int_428466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 43), tuple_428464, int_428466)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), tuple_428459, tuple_428464)
        
        # Testing the type of a for loop iterable (line 106)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 16), tuple_428459)
        # Getting the type of the for loop variable (line 106)
        for_loop_var_428467 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 16), tuple_428459)
        # Assigning a type to the variable 'axis' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'axis', for_loop_var_428467)
        # SSA begins for a for statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Call to spnorm(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'S' (line 107)
        S_428470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 43), 'S', False)
        # Processing the call keyword arguments (line 107)
        # Getting the type of 'axis' (line 107)
        axis_428471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 51), 'axis', False)
        keyword_428472 = axis_428471
        kwargs_428473 = {'axis': keyword_428472}
        # Getting the type of 'spnorm' (line 107)
        spnorm_428469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 107)
        spnorm_call_result_428474 = invoke(stypy.reporting.localization.Localization(__file__, 107, 36), spnorm_428469, *[S_428470], **kwargs_428473)
        
        
        # Call to npnorm(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'M' (line 107)
        M_428476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 65), 'M', False)
        # Processing the call keyword arguments (line 107)
        # Getting the type of 'axis' (line 107)
        axis_428477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 73), 'axis', False)
        keyword_428478 = axis_428477
        kwargs_428479 = {'axis': keyword_428478}
        # Getting the type of 'npnorm' (line 107)
        npnorm_428475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 58), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 107)
        npnorm_call_result_428480 = invoke(stypy.reporting.localization.Localization(__file__, 107, 58), npnorm_428475, *[M_428476], **kwargs_428479)
        
        # Processing the call keyword arguments (line 107)
        kwargs_428481 = {}
        # Getting the type of 'assert_allclose' (line 107)
        assert_allclose_428468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 107)
        assert_allclose_call_result_428482 = invoke(stypy.reporting.localization.Localization(__file__, 107, 20), assert_allclose_428468, *[spnorm_call_result_428474, npnorm_call_result_428480], **kwargs_428481)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_428483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        str_428484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'str', 'fro')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), tuple_428483, str_428484)
        # Adding element type (line 108)
        # Getting the type of 'np' (line 108)
        np_428485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 38), 'np')
        # Obtaining the member 'inf' of a type (line 108)
        inf_428486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 38), np_428485, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), tuple_428483, inf_428486)
        # Adding element type (line 108)
        
        # Getting the type of 'np' (line 108)
        np_428487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 47), 'np')
        # Obtaining the member 'inf' of a type (line 108)
        inf_428488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 47), np_428487, 'inf')
        # Applying the 'usub' unary operator (line 108)
        result___neg___428489 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 46), 'usub', inf_428488)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), tuple_428483, result___neg___428489)
        # Adding element type (line 108)
        int_428490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), tuple_428483, int_428490)
        # Adding element type (line 108)
        int_428491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), tuple_428483, int_428491)
        
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 20), tuple_428483)
        # Getting the type of the for loop variable (line 108)
        for_loop_var_428492 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 20), tuple_428483)
        # Assigning a type to the variable 'ord' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'ord', for_loop_var_428492)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Call to spnorm(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'S' (line 109)
        S_428495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 47), 'S', False)
        # Getting the type of 'ord' (line 109)
        ord_428496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 50), 'ord', False)
        # Processing the call keyword arguments (line 109)
        # Getting the type of 'axis' (line 109)
        axis_428497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 60), 'axis', False)
        keyword_428498 = axis_428497
        kwargs_428499 = {'axis': keyword_428498}
        # Getting the type of 'spnorm' (line 109)
        spnorm_428494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 109)
        spnorm_call_result_428500 = invoke(stypy.reporting.localization.Localization(__file__, 109, 40), spnorm_428494, *[S_428495, ord_428496], **kwargs_428499)
        
        
        # Call to npnorm(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'M' (line 110)
        M_428502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 47), 'M', False)
        # Getting the type of 'ord' (line 110)
        ord_428503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 50), 'ord', False)
        # Processing the call keyword arguments (line 110)
        # Getting the type of 'axis' (line 110)
        axis_428504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 60), 'axis', False)
        keyword_428505 = axis_428504
        kwargs_428506 = {'axis': keyword_428505}
        # Getting the type of 'npnorm' (line 110)
        npnorm_428501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 40), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 110)
        npnorm_call_result_428507 = invoke(stypy.reporting.localization.Localization(__file__, 110, 40), npnorm_428501, *[M_428502, ord_428503], **kwargs_428506)
        
        # Processing the call keyword arguments (line 109)
        kwargs_428508 = {}
        # Getting the type of 'assert_allclose' (line 109)
        assert_allclose_428493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 109)
        assert_allclose_call_result_428509 = invoke(stypy.reporting.localization.Localization(__file__, 109, 24), assert_allclose_428493, *[spnorm_call_result_428500, npnorm_call_result_428507], **kwargs_428508)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_428510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_428511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        int_428512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 29), tuple_428511, int_428512)
        # Adding element type (line 112)
        int_428513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 29), tuple_428511, int_428513)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), tuple_428510, tuple_428511)
        # Adding element type (line 112)
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_428514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        int_428515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 39), tuple_428514, int_428515)
        # Adding element type (line 112)
        int_428516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 39), tuple_428514, int_428516)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), tuple_428510, tuple_428514)
        # Adding element type (line 112)
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_428517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        int_428518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 49), tuple_428517, int_428518)
        # Adding element type (line 112)
        int_428519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 49), tuple_428517, int_428519)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), tuple_428510, tuple_428517)
        
        # Testing the type of a for loop iterable (line 112)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 16), tuple_428510)
        # Getting the type of the for loop variable (line 112)
        for_loop_var_428520 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 16), tuple_428510)
        # Assigning a type to the variable 'axis' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'axis', for_loop_var_428520)
        # SSA begins for a for statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to spnorm(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'S' (line 113)
        S_428523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 43), 'S', False)
        # Processing the call keyword arguments (line 113)
        # Getting the type of 'axis' (line 113)
        axis_428524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 51), 'axis', False)
        keyword_428525 = axis_428524
        kwargs_428526 = {'axis': keyword_428525}
        # Getting the type of 'spnorm' (line 113)
        spnorm_428522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 36), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 113)
        spnorm_call_result_428527 = invoke(stypy.reporting.localization.Localization(__file__, 113, 36), spnorm_428522, *[S_428523], **kwargs_428526)
        
        
        # Call to npnorm(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'M' (line 113)
        M_428529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 65), 'M', False)
        # Processing the call keyword arguments (line 113)
        # Getting the type of 'axis' (line 113)
        axis_428530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 73), 'axis', False)
        keyword_428531 = axis_428530
        kwargs_428532 = {'axis': keyword_428531}
        # Getting the type of 'npnorm' (line 113)
        npnorm_428528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 58), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 113)
        npnorm_call_result_428533 = invoke(stypy.reporting.localization.Localization(__file__, 113, 58), npnorm_428528, *[M_428529], **kwargs_428532)
        
        # Processing the call keyword arguments (line 113)
        kwargs_428534 = {}
        # Getting the type of 'assert_allclose' (line 113)
        assert_allclose_428521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 113)
        assert_allclose_call_result_428535 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), assert_allclose_428521, *[spnorm_call_result_428527, npnorm_call_result_428533], **kwargs_428534)
        
        
        # Call to assert_allclose(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to spnorm(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'S' (line 114)
        S_428538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 43), 'S', False)
        str_428539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'str', 'f')
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'axis' (line 114)
        axis_428540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 56), 'axis', False)
        keyword_428541 = axis_428540
        kwargs_428542 = {'axis': keyword_428541}
        # Getting the type of 'spnorm' (line 114)
        spnorm_428537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 36), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 114)
        spnorm_call_result_428543 = invoke(stypy.reporting.localization.Localization(__file__, 114, 36), spnorm_428537, *[S_428538, str_428539], **kwargs_428542)
        
        
        # Call to npnorm(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'M' (line 115)
        M_428545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 43), 'M', False)
        str_428546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 46), 'str', 'f')
        # Processing the call keyword arguments (line 115)
        # Getting the type of 'axis' (line 115)
        axis_428547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 56), 'axis', False)
        keyword_428548 = axis_428547
        kwargs_428549 = {'axis': keyword_428548}
        # Getting the type of 'npnorm' (line 115)
        npnorm_428544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 115)
        npnorm_call_result_428550 = invoke(stypy.reporting.localization.Localization(__file__, 115, 36), npnorm_428544, *[M_428545, str_428546], **kwargs_428549)
        
        # Processing the call keyword arguments (line 114)
        kwargs_428551 = {}
        # Getting the type of 'assert_allclose' (line 114)
        assert_allclose_428536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 114)
        assert_allclose_call_result_428552 = invoke(stypy.reporting.localization.Localization(__file__, 114, 20), assert_allclose_428536, *[spnorm_call_result_428543, npnorm_call_result_428550], **kwargs_428551)
        
        
        # Call to assert_allclose(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to spnorm(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'S' (line 116)
        S_428555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 43), 'S', False)
        str_428556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 46), 'str', 'fro')
        # Processing the call keyword arguments (line 116)
        # Getting the type of 'axis' (line 116)
        axis_428557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 58), 'axis', False)
        keyword_428558 = axis_428557
        kwargs_428559 = {'axis': keyword_428558}
        # Getting the type of 'spnorm' (line 116)
        spnorm_428554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 36), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 116)
        spnorm_call_result_428560 = invoke(stypy.reporting.localization.Localization(__file__, 116, 36), spnorm_428554, *[S_428555, str_428556], **kwargs_428559)
        
        
        # Call to npnorm(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'M' (line 117)
        M_428562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 43), 'M', False)
        str_428563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 46), 'str', 'fro')
        # Processing the call keyword arguments (line 117)
        # Getting the type of 'axis' (line 117)
        axis_428564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 58), 'axis', False)
        keyword_428565 = axis_428564
        kwargs_428566 = {'axis': keyword_428565}
        # Getting the type of 'npnorm' (line 117)
        npnorm_428561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 117)
        npnorm_call_result_428567 = invoke(stypy.reporting.localization.Localization(__file__, 117, 36), npnorm_428561, *[M_428562, str_428563], **kwargs_428566)
        
        # Processing the call keyword arguments (line 116)
        kwargs_428568 = {}
        # Getting the type of 'assert_allclose' (line 116)
        assert_allclose_428553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 116)
        assert_allclose_call_result_428569 = invoke(stypy.reporting.localization.Localization(__file__, 116, 20), assert_allclose_428553, *[spnorm_call_result_428560, npnorm_call_result_428567], **kwargs_428568)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sparse_matrix_norms_with_axis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_matrix_norms_with_axis' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_428570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428570)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_matrix_norms_with_axis'
        return stypy_return_type_428570


    @norecursion
    def test_sparse_vector_norms(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_vector_norms'
        module_type_store = module_type_store.open_function_context('test_sparse_vector_norms', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_localization', localization)
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_function_name', 'TestVsNumpyNorm.test_sparse_vector_norms')
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_param_names_list', [])
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVsNumpyNorm.test_sparse_vector_norms.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVsNumpyNorm.test_sparse_vector_norms', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_vector_norms', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_vector_norms(...)' code ##################

        
        # Getting the type of 'self' (line 120)
        self_428571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'self')
        # Obtaining the member '_sparse_types' of a type (line 120)
        _sparse_types_428572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), self_428571, '_sparse_types')
        # Testing the type of a for loop iterable (line 120)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 8), _sparse_types_428572)
        # Getting the type of the for loop variable (line 120)
        for_loop_var_428573 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 8), _sparse_types_428572)
        # Assigning a type to the variable 'sparse_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'sparse_type', for_loop_var_428573)
        # SSA begins for a for statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 121)
        self_428574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'self')
        # Obtaining the member '_test_matrices' of a type (line 121)
        _test_matrices_428575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 21), self_428574, '_test_matrices')
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 12), _test_matrices_428575)
        # Getting the type of the for loop variable (line 121)
        for_loop_var_428576 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 12), _test_matrices_428575)
        # Assigning a type to the variable 'M' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'M', for_loop_var_428576)
        # SSA begins for a for statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 122):
        
        # Call to sparse_type(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'M' (line 122)
        M_428578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'M', False)
        # Processing the call keyword arguments (line 122)
        kwargs_428579 = {}
        # Getting the type of 'sparse_type' (line 122)
        sparse_type_428577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'sparse_type', False)
        # Calling sparse_type(args, kwargs) (line 122)
        sparse_type_call_result_428580 = invoke(stypy.reporting.localization.Localization(__file__, 122, 20), sparse_type_428577, *[M_428578], **kwargs_428579)
        
        # Assigning a type to the variable 'S' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'S', sparse_type_call_result_428580)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_428581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        int_428582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), tuple_428581, int_428582)
        # Adding element type (line 123)
        int_428583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), tuple_428581, int_428583)
        # Adding element type (line 123)
        int_428584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), tuple_428581, int_428584)
        # Adding element type (line 123)
        int_428585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), tuple_428581, int_428585)
        # Adding element type (line 123)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_428586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        int_428587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 44), tuple_428586, int_428587)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), tuple_428581, tuple_428586)
        # Adding element type (line 123)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_428588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        int_428589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 51), tuple_428588, int_428589)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), tuple_428581, tuple_428588)
        # Adding element type (line 123)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_428590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        int_428591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 58), tuple_428590, int_428591)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), tuple_428581, tuple_428590)
        # Adding element type (line 123)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_428592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        int_428593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 66), tuple_428592, int_428593)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), tuple_428581, tuple_428592)
        
        # Testing the type of a for loop iterable (line 123)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 16), tuple_428581)
        # Getting the type of the for loop variable (line 123)
        for_loop_var_428594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 16), tuple_428581)
        # Assigning a type to the variable 'axis' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'axis', for_loop_var_428594)
        # SSA begins for a for statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Call to spnorm(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'S' (line 124)
        S_428597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 43), 'S', False)
        # Processing the call keyword arguments (line 124)
        # Getting the type of 'axis' (line 124)
        axis_428598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 51), 'axis', False)
        keyword_428599 = axis_428598
        kwargs_428600 = {'axis': keyword_428599}
        # Getting the type of 'spnorm' (line 124)
        spnorm_428596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 124)
        spnorm_call_result_428601 = invoke(stypy.reporting.localization.Localization(__file__, 124, 36), spnorm_428596, *[S_428597], **kwargs_428600)
        
        
        # Call to npnorm(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'M' (line 124)
        M_428603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 65), 'M', False)
        # Processing the call keyword arguments (line 124)
        # Getting the type of 'axis' (line 124)
        axis_428604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 73), 'axis', False)
        keyword_428605 = axis_428604
        kwargs_428606 = {'axis': keyword_428605}
        # Getting the type of 'npnorm' (line 124)
        npnorm_428602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 58), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 124)
        npnorm_call_result_428607 = invoke(stypy.reporting.localization.Localization(__file__, 124, 58), npnorm_428602, *[M_428603], **kwargs_428606)
        
        # Processing the call keyword arguments (line 124)
        kwargs_428608 = {}
        # Getting the type of 'assert_allclose' (line 124)
        assert_allclose_428595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 124)
        assert_allclose_call_result_428609 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), assert_allclose_428595, *[spnorm_call_result_428601, npnorm_call_result_428607], **kwargs_428608)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_428610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        # Getting the type of 'None' (line 125)
        None_428611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 31), tuple_428610, None_428611)
        # Adding element type (line 125)
        int_428612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 31), tuple_428610, int_428612)
        # Adding element type (line 125)
        # Getting the type of 'np' (line 125)
        np_428613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'np')
        # Obtaining the member 'inf' of a type (line 125)
        inf_428614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 40), np_428613, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 31), tuple_428610, inf_428614)
        # Adding element type (line 125)
        
        # Getting the type of 'np' (line 125)
        np_428615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 49), 'np')
        # Obtaining the member 'inf' of a type (line 125)
        inf_428616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 49), np_428615, 'inf')
        # Applying the 'usub' unary operator (line 125)
        result___neg___428617 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 48), 'usub', inf_428616)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 31), tuple_428610, result___neg___428617)
        # Adding element type (line 125)
        int_428618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 31), tuple_428610, int_428618)
        # Adding element type (line 125)
        float_428619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 31), tuple_428610, float_428619)
        # Adding element type (line 125)
        float_428620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 31), tuple_428610, float_428620)
        
        # Testing the type of a for loop iterable (line 125)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 125, 20), tuple_428610)
        # Getting the type of the for loop variable (line 125)
        for_loop_var_428621 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 125, 20), tuple_428610)
        # Assigning a type to the variable 'ord' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'ord', for_loop_var_428621)
        # SSA begins for a for statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to spnorm(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'S' (line 126)
        S_428624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 47), 'S', False)
        # Getting the type of 'ord' (line 126)
        ord_428625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 50), 'ord', False)
        # Processing the call keyword arguments (line 126)
        # Getting the type of 'axis' (line 126)
        axis_428626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 60), 'axis', False)
        keyword_428627 = axis_428626
        kwargs_428628 = {'axis': keyword_428627}
        # Getting the type of 'spnorm' (line 126)
        spnorm_428623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 40), 'spnorm', False)
        # Calling spnorm(args, kwargs) (line 126)
        spnorm_call_result_428629 = invoke(stypy.reporting.localization.Localization(__file__, 126, 40), spnorm_428623, *[S_428624, ord_428625], **kwargs_428628)
        
        
        # Call to npnorm(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'M' (line 127)
        M_428631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'M', False)
        # Getting the type of 'ord' (line 127)
        ord_428632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 50), 'ord', False)
        # Processing the call keyword arguments (line 127)
        # Getting the type of 'axis' (line 127)
        axis_428633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 60), 'axis', False)
        keyword_428634 = axis_428633
        kwargs_428635 = {'axis': keyword_428634}
        # Getting the type of 'npnorm' (line 127)
        npnorm_428630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 40), 'npnorm', False)
        # Calling npnorm(args, kwargs) (line 127)
        npnorm_call_result_428636 = invoke(stypy.reporting.localization.Localization(__file__, 127, 40), npnorm_428630, *[M_428631, ord_428632], **kwargs_428635)
        
        # Processing the call keyword arguments (line 126)
        kwargs_428637 = {}
        # Getting the type of 'assert_allclose' (line 126)
        assert_allclose_428622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 126)
        assert_allclose_call_result_428638 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), assert_allclose_428622, *[spnorm_call_result_428629, npnorm_call_result_428636], **kwargs_428637)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sparse_vector_norms(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_vector_norms' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_428639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428639)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_vector_norms'
        return stypy_return_type_428639


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 71, 0, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVsNumpyNorm.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestVsNumpyNorm' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'TestVsNumpyNorm', TestVsNumpyNorm)

# Assigning a Tuple to a Name (line 72):

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_428640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
# Getting the type of 'scipy' (line 73)
scipy_428641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'scipy')
# Obtaining the member 'sparse' of a type (line 73)
sparse_428642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), scipy_428641, 'sparse')
# Obtaining the member 'bsr_matrix' of a type (line 73)
bsr_matrix_428643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), sparse_428642, 'bsr_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_428640, bsr_matrix_428643)
# Adding element type (line 73)
# Getting the type of 'scipy' (line 74)
scipy_428644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'scipy')
# Obtaining the member 'sparse' of a type (line 74)
sparse_428645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), scipy_428644, 'sparse')
# Obtaining the member 'coo_matrix' of a type (line 74)
coo_matrix_428646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), sparse_428645, 'coo_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_428640, coo_matrix_428646)
# Adding element type (line 73)
# Getting the type of 'scipy' (line 75)
scipy_428647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'scipy')
# Obtaining the member 'sparse' of a type (line 75)
sparse_428648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), scipy_428647, 'sparse')
# Obtaining the member 'csc_matrix' of a type (line 75)
csc_matrix_428649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), sparse_428648, 'csc_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_428640, csc_matrix_428649)
# Adding element type (line 73)
# Getting the type of 'scipy' (line 76)
scipy_428650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'scipy')
# Obtaining the member 'sparse' of a type (line 76)
sparse_428651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), scipy_428650, 'sparse')
# Obtaining the member 'csr_matrix' of a type (line 76)
csr_matrix_428652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), sparse_428651, 'csr_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_428640, csr_matrix_428652)
# Adding element type (line 73)
# Getting the type of 'scipy' (line 77)
scipy_428653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'scipy')
# Obtaining the member 'sparse' of a type (line 77)
sparse_428654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), scipy_428653, 'sparse')
# Obtaining the member 'dia_matrix' of a type (line 77)
dia_matrix_428655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), sparse_428654, 'dia_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_428640, dia_matrix_428655)
# Adding element type (line 73)
# Getting the type of 'scipy' (line 78)
scipy_428656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'scipy')
# Obtaining the member 'sparse' of a type (line 78)
sparse_428657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), scipy_428656, 'sparse')
# Obtaining the member 'dok_matrix' of a type (line 78)
dok_matrix_428658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), sparse_428657, 'dok_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_428640, dok_matrix_428658)
# Adding element type (line 73)
# Getting the type of 'scipy' (line 79)
scipy_428659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'scipy')
# Obtaining the member 'sparse' of a type (line 79)
sparse_428660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), scipy_428659, 'sparse')
# Obtaining the member 'lil_matrix' of a type (line 79)
lil_matrix_428661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), sparse_428660, 'lil_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_428640, lil_matrix_428661)

# Getting the type of 'TestVsNumpyNorm'
TestVsNumpyNorm_428662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestVsNumpyNorm')
# Setting the type of the member '_sparse_types' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestVsNumpyNorm_428662, '_sparse_types', tuple_428640)

# Assigning a Tuple to a Name (line 81):

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_428663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)

# Call to reshape(...): (line 82)
# Processing the call arguments (line 82)

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_428672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 40), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
int_428673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 40), tuple_428672, int_428673)
# Adding element type (line 82)
int_428674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 40), tuple_428672, int_428674)

# Processing the call keyword arguments (line 82)
kwargs_428675 = {}

# Call to arange(...): (line 82)
# Processing the call arguments (line 82)
int_428666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'int')
# Processing the call keyword arguments (line 82)
kwargs_428667 = {}
# Getting the type of 'np' (line 82)
np_428664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'np', False)
# Obtaining the member 'arange' of a type (line 82)
arange_428665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), np_428664, 'arange')
# Calling arange(args, kwargs) (line 82)
arange_call_result_428668 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), arange_428665, *[int_428666], **kwargs_428667)

int_428669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 28), 'int')
# Applying the binary operator '-' (line 82)
result_sub_428670 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 13), '-', arange_call_result_428668, int_428669)

# Obtaining the member 'reshape' of a type (line 82)
reshape_428671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), result_sub_428670, 'reshape')
# Calling reshape(args, kwargs) (line 82)
reshape_call_result_428676 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), reshape_428671, *[tuple_428672], **kwargs_428675)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), tuple_428663, reshape_call_result_428676)
# Adding element type (line 82)

# Obtaining an instance of the builtin type 'list' (line 83)
list_428677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 83)
# Adding element type (line 83)

# Obtaining an instance of the builtin type 'list' (line 84)
list_428678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 84)
# Adding element type (line 84)
int_428679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), list_428678, int_428679)
# Adding element type (line 84)
int_428680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), list_428678, int_428680)
# Adding element type (line 84)
int_428681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), list_428678, int_428681)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 12), list_428677, list_428678)
# Adding element type (line 83)

# Obtaining an instance of the builtin type 'list' (line 85)
list_428682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 85)
# Adding element type (line 85)
int_428683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 16), list_428682, int_428683)
# Adding element type (line 85)
int_428684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 16), list_428682, int_428684)
# Adding element type (line 85)
int_428685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 16), list_428682, int_428685)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 12), list_428677, list_428682)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), tuple_428663, list_428677)
# Adding element type (line 82)

# Obtaining an instance of the builtin type 'list' (line 86)
list_428686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 86)
# Adding element type (line 86)

# Obtaining an instance of the builtin type 'list' (line 87)
list_428687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 87)
# Adding element type (line 87)
int_428688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 16), list_428687, int_428688)
# Adding element type (line 87)
int_428689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 16), list_428687, int_428689)
# Adding element type (line 87)
int_428690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 16), list_428687, int_428690)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 12), list_428686, list_428687)
# Adding element type (line 86)

# Obtaining an instance of the builtin type 'list' (line 88)
list_428691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 88)
# Adding element type (line 88)
int_428692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 16), list_428691, int_428692)
# Adding element type (line 88)
int_428693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 16), list_428691, int_428693)
# Adding element type (line 88)
complex_428694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 16), list_428691, complex_428694)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 12), list_428686, list_428691)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), tuple_428663, list_428686)

# Getting the type of 'TestVsNumpyNorm'
TestVsNumpyNorm_428695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestVsNumpyNorm')
# Setting the type of the member '_test_matrices' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestVsNumpyNorm_428695, '_test_matrices', tuple_428663)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
