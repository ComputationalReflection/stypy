
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import numpy as np
2: from numpy.linalg import lstsq
3: from numpy.testing import assert_allclose, assert_equal, assert_
4: from pytest import raises as assert_raises
5: 
6: from scipy.sparse import rand
7: from scipy.sparse.linalg import aslinearoperator
8: from scipy.optimize import lsq_linear
9: 
10: 
11: A = np.array([
12:     [0.171, -0.057],
13:     [-0.049, -0.248],
14:     [-0.166, 0.054],
15: ])
16: b = np.array([0.074, 1.014, -0.383])
17: 
18: 
19: class BaseMixin(object):
20:     def setup_method(self):
21:         self.rnd = np.random.RandomState(0)
22: 
23:     def test_dense_no_bounds(self):
24:         for lsq_solver in self.lsq_solvers:
25:             res = lsq_linear(A, b, method=self.method, lsq_solver=lsq_solver)
26:             assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])
27: 
28:     def test_dense_bounds(self):
29:         # Solutions for comparison are taken from MATLAB.
30:         lb = np.array([-1, -10])
31:         ub = np.array([1, 0])
32:         for lsq_solver in self.lsq_solvers:
33:             res = lsq_linear(A, b, (lb, ub), method=self.method,
34:                              lsq_solver=lsq_solver)
35:             assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])
36: 
37:         lb = np.array([0.0, -np.inf])
38:         for lsq_solver in self.lsq_solvers:
39:             res = lsq_linear(A, b, (lb, np.inf), method=self.method,
40:                              lsq_solver=lsq_solver)
41:             assert_allclose(res.x, np.array([0.0, -4.084174437334673]),
42:                             atol=1e-6)
43: 
44:         lb = np.array([-1, 0])
45:         for lsq_solver in self.lsq_solvers:
46:             res = lsq_linear(A, b, (lb, np.inf), method=self.method,
47:                              lsq_solver=lsq_solver)
48:             assert_allclose(res.x, np.array([0.448427311733504, 0]),
49:                             atol=1e-15)
50: 
51:         ub = np.array([np.inf, -5])
52:         for lsq_solver in self.lsq_solvers:
53:             res = lsq_linear(A, b, (-np.inf, ub), method=self.method,
54:                              lsq_solver=lsq_solver)
55:             assert_allclose(res.x, np.array([-0.105560998682388, -5]))
56: 
57:         ub = np.array([-1, np.inf])
58:         for lsq_solver in self.lsq_solvers:
59:             res = lsq_linear(A, b, (-np.inf, ub), method=self.method,
60:                              lsq_solver=lsq_solver)
61:             assert_allclose(res.x, np.array([-1, -4.181102129483254]))
62: 
63:         lb = np.array([0, -4])
64:         ub = np.array([1, 0])
65:         for lsq_solver in self.lsq_solvers:
66:             res = lsq_linear(A, b, (lb, ub), method=self.method,
67:                              lsq_solver=lsq_solver)
68:             assert_allclose(res.x, np.array([0.005236663400791, -4]))
69: 
70:     def test_dense_rank_deficient(self):
71:         A = np.array([[-0.307, -0.184]])
72:         b = np.array([0.773])
73:         lb = [-0.1, -0.1]
74:         ub = [0.1, 0.1]
75:         for lsq_solver in self.lsq_solvers:
76:             res = lsq_linear(A, b, (lb, ub), method=self.method,
77:                              lsq_solver=lsq_solver)
78:             assert_allclose(res.x, [-0.1, -0.1])
79: 
80:         A = np.array([
81:             [0.334, 0.668],
82:             [-0.516, -1.032],
83:             [0.192, 0.384],
84:         ])
85:         b = np.array([-1.436, 0.135, 0.909])
86:         lb = [0, -1]
87:         ub = [1, -0.5]
88:         for lsq_solver in self.lsq_solvers:
89:             res = lsq_linear(A, b, (lb, ub), method=self.method,
90:                              lsq_solver=lsq_solver)
91:             assert_allclose(res.optimality, 0, atol=1e-11)
92: 
93:     def test_full_result(self):
94:         lb = np.array([0, -4])
95:         ub = np.array([1, 0])
96:         res = lsq_linear(A, b, (lb, ub), method=self.method)
97: 
98:         assert_allclose(res.x, [0.005236663400791, -4])
99: 
100:         r = A.dot(res.x) - b
101:         assert_allclose(res.cost, 0.5 * np.dot(r, r))
102:         assert_allclose(res.fun, r)
103: 
104:         assert_allclose(res.optimality, 0.0, atol=1e-12)
105:         assert_equal(res.active_mask, [0, -1])
106:         assert_(res.nit < 15)
107:         assert_(res.status == 1 or res.status == 3)
108:         assert_(isinstance(res.message, str))
109:         assert_(res.success)
110: 
111: 
112: class SparseMixin(object):
113:     def test_sparse_and_LinearOperator(self):
114:         m = 5000
115:         n = 1000
116:         A = rand(m, n, random_state=0)
117:         b = self.rnd.randn(m)
118:         res = lsq_linear(A, b)
119:         assert_allclose(res.optimality, 0, atol=1e-6)
120: 
121:         A = aslinearoperator(A)
122:         res = lsq_linear(A, b)
123:         assert_allclose(res.optimality, 0, atol=1e-6)
124: 
125:     def test_sparse_bounds(self):
126:         m = 5000
127:         n = 1000
128:         A = rand(m, n, random_state=0)
129:         b = self.rnd.randn(m)
130:         lb = self.rnd.randn(n)
131:         ub = lb + 1
132:         res = lsq_linear(A, b, (lb, ub))
133:         assert_allclose(res.optimality, 0.0, atol=1e-8)
134: 
135:         res = lsq_linear(A, b, (lb, ub), lsmr_tol=1e-13)
136:         assert_allclose(res.optimality, 0.0, atol=1e-8)
137: 
138:         res = lsq_linear(A, b, (lb, ub), lsmr_tol='auto')
139:         assert_allclose(res.optimality, 0.0, atol=1e-8)
140: 
141: 
142: class TestTRF(BaseMixin, SparseMixin):
143:     method = 'trf'
144:     lsq_solvers = ['exact', 'lsmr']
145: 
146: 
147: class TestBVLS(BaseMixin):
148:     method = 'bvls'
149:     lsq_solvers = ['exact']
150: 
151: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import numpy' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_217326 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy')

if (type(import_217326) is not StypyTypeError):

    if (import_217326 != 'pyd_module'):
        __import__(import_217326)
        sys_modules_217327 = sys.modules[import_217326]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'np', sys_modules_217327.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', import_217326)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from numpy.linalg import lstsq' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_217328 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.linalg')

if (type(import_217328) is not StypyTypeError):

    if (import_217328 != 'pyd_module'):
        __import__(import_217328)
        sys_modules_217329 = sys.modules[import_217328]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.linalg', sys_modules_217329.module_type_store, module_type_store, ['lstsq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_217329, sys_modules_217329.module_type_store, module_type_store)
    else:
        from numpy.linalg import lstsq

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.linalg', None, module_type_store, ['lstsq'], [lstsq])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.linalg', import_217328)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_allclose, assert_equal, assert_' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_217330 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_217330) is not StypyTypeError):

    if (import_217330 != 'pyd_module'):
        __import__(import_217330)
        sys_modules_217331 = sys.modules[import_217330]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_217331.module_type_store, module_type_store, ['assert_allclose', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_217331, sys_modules_217331.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal', 'assert_'], [assert_allclose, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_217330)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from pytest import assert_raises' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_217332 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_217332) is not StypyTypeError):

    if (import_217332 != 'pyd_module'):
        __import__(import_217332)
        sys_modules_217333 = sys.modules[import_217332]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_217333.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_217333, sys_modules_217333.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_217332)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse import rand' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_217334 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse')

if (type(import_217334) is not StypyTypeError):

    if (import_217334 != 'pyd_module'):
        __import__(import_217334)
        sys_modules_217335 = sys.modules[import_217334]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', sys_modules_217335.module_type_store, module_type_store, ['rand'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_217335, sys_modules_217335.module_type_store, module_type_store)
    else:
        from scipy.sparse import rand

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', None, module_type_store, ['rand'], [rand])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', import_217334)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse.linalg import aslinearoperator' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_217336 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg')

if (type(import_217336) is not StypyTypeError):

    if (import_217336 != 'pyd_module'):
        __import__(import_217336)
        sys_modules_217337 = sys.modules[import_217336]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', sys_modules_217337.module_type_store, module_type_store, ['aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_217337, sys_modules_217337.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', import_217336)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize import lsq_linear' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_217338 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize')

if (type(import_217338) is not StypyTypeError):

    if (import_217338 != 'pyd_module'):
        __import__(import_217338)
        sys_modules_217339 = sys.modules[import_217338]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', sys_modules_217339.module_type_store, module_type_store, ['lsq_linear'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_217339, sys_modules_217339.module_type_store, module_type_store)
    else:
        from scipy.optimize import lsq_linear

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', None, module_type_store, ['lsq_linear'], [lsq_linear])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', import_217338)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


# Assigning a Call to a Name (line 11):

# Call to array(...): (line 11)
# Processing the call arguments (line 11)

# Obtaining an instance of the builtin type 'list' (line 11)
list_217342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'list' (line 12)
list_217343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
float_217344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), list_217343, float_217344)
# Adding element type (line 12)
float_217345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), list_217343, float_217345)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_217342, list_217343)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'list' (line 13)
list_217346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
float_217347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_217346, float_217347)
# Adding element type (line 13)
float_217348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 13), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_217346, float_217348)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_217342, list_217346)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'list' (line 14)
list_217349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
float_217350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_217349, float_217350)
# Adding element type (line 14)
float_217351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 13), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_217349, float_217351)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_217342, list_217349)

# Processing the call keyword arguments (line 11)
kwargs_217352 = {}
# Getting the type of 'np' (line 11)
np_217340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'array' of a type (line 11)
array_217341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_217340, 'array')
# Calling array(args, kwargs) (line 11)
array_call_result_217353 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), array_217341, *[list_217342], **kwargs_217352)

# Assigning a type to the variable 'A' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'A', array_call_result_217353)

# Assigning a Call to a Name (line 16):

# Call to array(...): (line 16)
# Processing the call arguments (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_217356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
float_217357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), list_217356, float_217357)
# Adding element type (line 16)
float_217358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), list_217356, float_217358)
# Adding element type (line 16)
float_217359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), list_217356, float_217359)

# Processing the call keyword arguments (line 16)
kwargs_217360 = {}
# Getting the type of 'np' (line 16)
np_217354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'np', False)
# Obtaining the member 'array' of a type (line 16)
array_217355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), np_217354, 'array')
# Calling array(args, kwargs) (line 16)
array_call_result_217361 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), array_217355, *[list_217356], **kwargs_217360)

# Assigning a type to the variable 'b' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'b', array_call_result_217361)
# Declaration of the 'BaseMixin' class

class BaseMixin(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.setup_method.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.setup_method.__dict__.__setitem__('stypy_function_name', 'BaseMixin.setup_method')
        BaseMixin.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 21):
        
        # Call to RandomState(...): (line 21)
        # Processing the call arguments (line 21)
        int_217365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 41), 'int')
        # Processing the call keyword arguments (line 21)
        kwargs_217366 = {}
        # Getting the type of 'np' (line 21)
        np_217362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'np', False)
        # Obtaining the member 'random' of a type (line 21)
        random_217363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 19), np_217362, 'random')
        # Obtaining the member 'RandomState' of a type (line 21)
        RandomState_217364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 19), random_217363, 'RandomState')
        # Calling RandomState(args, kwargs) (line 21)
        RandomState_call_result_217367 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), RandomState_217364, *[int_217365], **kwargs_217366)
        
        # Getting the type of 'self' (line 21)
        self_217368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member 'rnd' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_217368, 'rnd', RandomState_call_result_217367)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_217369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_217369


    @norecursion
    def test_dense_no_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dense_no_bounds'
        module_type_store = module_type_store.open_function_context('test_dense_no_bounds', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_dense_no_bounds')
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_dense_no_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_dense_no_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dense_no_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dense_no_bounds(...)' code ##################

        
        # Getting the type of 'self' (line 24)
        self_217370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 24)
        lsq_solvers_217371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 26), self_217370, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 24)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 24, 8), lsq_solvers_217371)
        # Getting the type of the for loop variable (line 24)
        for_loop_var_217372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 24, 8), lsq_solvers_217371)
        # Assigning a type to the variable 'lsq_solver' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'lsq_solver', for_loop_var_217372)
        # SSA begins for a for statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 25):
        
        # Call to lsq_linear(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'A' (line 25)
        A_217374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'A', False)
        # Getting the type of 'b' (line 25)
        b_217375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'b', False)
        # Processing the call keyword arguments (line 25)
        # Getting the type of 'self' (line 25)
        self_217376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 42), 'self', False)
        # Obtaining the member 'method' of a type (line 25)
        method_217377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 42), self_217376, 'method')
        keyword_217378 = method_217377
        # Getting the type of 'lsq_solver' (line 25)
        lsq_solver_217379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 66), 'lsq_solver', False)
        keyword_217380 = lsq_solver_217379
        kwargs_217381 = {'lsq_solver': keyword_217380, 'method': keyword_217378}
        # Getting the type of 'lsq_linear' (line 25)
        lsq_linear_217373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 25)
        lsq_linear_call_result_217382 = invoke(stypy.reporting.localization.Localization(__file__, 25, 18), lsq_linear_217373, *[A_217374, b_217375], **kwargs_217381)
        
        # Assigning a type to the variable 'res' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'res', lsq_linear_call_result_217382)
        
        # Call to assert_allclose(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'res' (line 26)
        res_217384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 26)
        x_217385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 28), res_217384, 'x')
        
        # Obtaining the type of the subscript
        int_217386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 57), 'int')
        
        # Call to lstsq(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'A' (line 26)
        A_217388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 41), 'A', False)
        # Getting the type of 'b' (line 26)
        b_217389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 44), 'b', False)
        # Processing the call keyword arguments (line 26)
        int_217390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 53), 'int')
        keyword_217391 = int_217390
        kwargs_217392 = {'rcond': keyword_217391}
        # Getting the type of 'lstsq' (line 26)
        lstsq_217387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'lstsq', False)
        # Calling lstsq(args, kwargs) (line 26)
        lstsq_call_result_217393 = invoke(stypy.reporting.localization.Localization(__file__, 26, 35), lstsq_217387, *[A_217388, b_217389], **kwargs_217392)
        
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___217394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 35), lstsq_call_result_217393, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_217395 = invoke(stypy.reporting.localization.Localization(__file__, 26, 35), getitem___217394, int_217386)
        
        # Processing the call keyword arguments (line 26)
        kwargs_217396 = {}
        # Getting the type of 'assert_allclose' (line 26)
        assert_allclose_217383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 26)
        assert_allclose_call_result_217397 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), assert_allclose_217383, *[x_217385, subscript_call_result_217395], **kwargs_217396)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dense_no_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dense_no_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_217398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217398)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dense_no_bounds'
        return stypy_return_type_217398


    @norecursion
    def test_dense_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dense_bounds'
        module_type_store = module_type_store.open_function_context('test_dense_bounds', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_dense_bounds')
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_dense_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_dense_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dense_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dense_bounds(...)' code ##################

        
        # Assigning a Call to a Name (line 30):
        
        # Call to array(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_217401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        int_217402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 22), list_217401, int_217402)
        # Adding element type (line 30)
        int_217403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 22), list_217401, int_217403)
        
        # Processing the call keyword arguments (line 30)
        kwargs_217404 = {}
        # Getting the type of 'np' (line 30)
        np_217399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 30)
        array_217400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 13), np_217399, 'array')
        # Calling array(args, kwargs) (line 30)
        array_call_result_217405 = invoke(stypy.reporting.localization.Localization(__file__, 30, 13), array_217400, *[list_217401], **kwargs_217404)
        
        # Assigning a type to the variable 'lb' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'lb', array_call_result_217405)
        
        # Assigning a Call to a Name (line 31):
        
        # Call to array(...): (line 31)
        # Processing the call arguments (line 31)
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_217408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_217409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 22), list_217408, int_217409)
        # Adding element type (line 31)
        int_217410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 22), list_217408, int_217410)
        
        # Processing the call keyword arguments (line 31)
        kwargs_217411 = {}
        # Getting the type of 'np' (line 31)
        np_217406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 31)
        array_217407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), np_217406, 'array')
        # Calling array(args, kwargs) (line 31)
        array_call_result_217412 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), array_217407, *[list_217408], **kwargs_217411)
        
        # Assigning a type to the variable 'ub' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'ub', array_call_result_217412)
        
        # Getting the type of 'self' (line 32)
        self_217413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 32)
        lsq_solvers_217414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 26), self_217413, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 32)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 8), lsq_solvers_217414)
        # Getting the type of the for loop variable (line 32)
        for_loop_var_217415 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 8), lsq_solvers_217414)
        # Assigning a type to the variable 'lsq_solver' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'lsq_solver', for_loop_var_217415)
        # SSA begins for a for statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 33):
        
        # Call to lsq_linear(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'A' (line 33)
        A_217417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'A', False)
        # Getting the type of 'b' (line 33)
        b_217418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 32), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_217419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        # Getting the type of 'lb' (line 33)
        lb_217420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 36), tuple_217419, lb_217420)
        # Adding element type (line 33)
        # Getting the type of 'ub' (line 33)
        ub_217421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 36), tuple_217419, ub_217421)
        
        # Processing the call keyword arguments (line 33)
        # Getting the type of 'self' (line 33)
        self_217422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 52), 'self', False)
        # Obtaining the member 'method' of a type (line 33)
        method_217423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 52), self_217422, 'method')
        keyword_217424 = method_217423
        # Getting the type of 'lsq_solver' (line 34)
        lsq_solver_217425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 40), 'lsq_solver', False)
        keyword_217426 = lsq_solver_217425
        kwargs_217427 = {'lsq_solver': keyword_217426, 'method': keyword_217424}
        # Getting the type of 'lsq_linear' (line 33)
        lsq_linear_217416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 33)
        lsq_linear_call_result_217428 = invoke(stypy.reporting.localization.Localization(__file__, 33, 18), lsq_linear_217416, *[A_217417, b_217418, tuple_217419], **kwargs_217427)
        
        # Assigning a type to the variable 'res' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'res', lsq_linear_call_result_217428)
        
        # Call to assert_allclose(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'res' (line 35)
        res_217430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 35)
        x_217431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 28), res_217430, 'x')
        
        # Obtaining the type of the subscript
        int_217432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 57), 'int')
        
        # Call to lstsq(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'A' (line 35)
        A_217434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 41), 'A', False)
        # Getting the type of 'b' (line 35)
        b_217435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 44), 'b', False)
        # Processing the call keyword arguments (line 35)
        int_217436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 53), 'int')
        keyword_217437 = int_217436
        kwargs_217438 = {'rcond': keyword_217437}
        # Getting the type of 'lstsq' (line 35)
        lstsq_217433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 35), 'lstsq', False)
        # Calling lstsq(args, kwargs) (line 35)
        lstsq_call_result_217439 = invoke(stypy.reporting.localization.Localization(__file__, 35, 35), lstsq_217433, *[A_217434, b_217435], **kwargs_217438)
        
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___217440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 35), lstsq_call_result_217439, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_217441 = invoke(stypy.reporting.localization.Localization(__file__, 35, 35), getitem___217440, int_217432)
        
        # Processing the call keyword arguments (line 35)
        kwargs_217442 = {}
        # Getting the type of 'assert_allclose' (line 35)
        assert_allclose_217429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 35)
        assert_allclose_call_result_217443 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), assert_allclose_217429, *[x_217431, subscript_call_result_217441], **kwargs_217442)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 37):
        
        # Call to array(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_217446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        # Adding element type (line 37)
        float_217447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), list_217446, float_217447)
        # Adding element type (line 37)
        
        # Getting the type of 'np' (line 37)
        np_217448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'np', False)
        # Obtaining the member 'inf' of a type (line 37)
        inf_217449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 29), np_217448, 'inf')
        # Applying the 'usub' unary operator (line 37)
        result___neg___217450 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 28), 'usub', inf_217449)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), list_217446, result___neg___217450)
        
        # Processing the call keyword arguments (line 37)
        kwargs_217451 = {}
        # Getting the type of 'np' (line 37)
        np_217444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 37)
        array_217445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), np_217444, 'array')
        # Calling array(args, kwargs) (line 37)
        array_call_result_217452 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), array_217445, *[list_217446], **kwargs_217451)
        
        # Assigning a type to the variable 'lb' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'lb', array_call_result_217452)
        
        # Getting the type of 'self' (line 38)
        self_217453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 38)
        lsq_solvers_217454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 26), self_217453, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), lsq_solvers_217454)
        # Getting the type of the for loop variable (line 38)
        for_loop_var_217455 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), lsq_solvers_217454)
        # Assigning a type to the variable 'lsq_solver' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'lsq_solver', for_loop_var_217455)
        # SSA begins for a for statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 39):
        
        # Call to lsq_linear(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'A' (line 39)
        A_217457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'A', False)
        # Getting the type of 'b' (line 39)
        b_217458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 32), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_217459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        # Getting the type of 'lb' (line 39)
        lb_217460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 36), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), tuple_217459, lb_217460)
        # Adding element type (line 39)
        # Getting the type of 'np' (line 39)
        np_217461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 40), 'np', False)
        # Obtaining the member 'inf' of a type (line 39)
        inf_217462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 40), np_217461, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), tuple_217459, inf_217462)
        
        # Processing the call keyword arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_217463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 56), 'self', False)
        # Obtaining the member 'method' of a type (line 39)
        method_217464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 56), self_217463, 'method')
        keyword_217465 = method_217464
        # Getting the type of 'lsq_solver' (line 40)
        lsq_solver_217466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 40), 'lsq_solver', False)
        keyword_217467 = lsq_solver_217466
        kwargs_217468 = {'lsq_solver': keyword_217467, 'method': keyword_217465}
        # Getting the type of 'lsq_linear' (line 39)
        lsq_linear_217456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 39)
        lsq_linear_call_result_217469 = invoke(stypy.reporting.localization.Localization(__file__, 39, 18), lsq_linear_217456, *[A_217457, b_217458, tuple_217459], **kwargs_217468)
        
        # Assigning a type to the variable 'res' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'res', lsq_linear_call_result_217469)
        
        # Call to assert_allclose(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'res' (line 41)
        res_217471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 41)
        x_217472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 28), res_217471, 'x')
        
        # Call to array(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_217475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        float_217476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 44), list_217475, float_217476)
        # Adding element type (line 41)
        float_217477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 44), list_217475, float_217477)
        
        # Processing the call keyword arguments (line 41)
        kwargs_217478 = {}
        # Getting the type of 'np' (line 41)
        np_217473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 41)
        array_217474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 35), np_217473, 'array')
        # Calling array(args, kwargs) (line 41)
        array_call_result_217479 = invoke(stypy.reporting.localization.Localization(__file__, 41, 35), array_217474, *[list_217475], **kwargs_217478)
        
        # Processing the call keyword arguments (line 41)
        float_217480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'float')
        keyword_217481 = float_217480
        kwargs_217482 = {'atol': keyword_217481}
        # Getting the type of 'assert_allclose' (line 41)
        assert_allclose_217470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 41)
        assert_allclose_call_result_217483 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), assert_allclose_217470, *[x_217472, array_call_result_217479], **kwargs_217482)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 44):
        
        # Call to array(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_217486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        int_217487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), list_217486, int_217487)
        # Adding element type (line 44)
        int_217488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), list_217486, int_217488)
        
        # Processing the call keyword arguments (line 44)
        kwargs_217489 = {}
        # Getting the type of 'np' (line 44)
        np_217484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 44)
        array_217485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), np_217484, 'array')
        # Calling array(args, kwargs) (line 44)
        array_call_result_217490 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), array_217485, *[list_217486], **kwargs_217489)
        
        # Assigning a type to the variable 'lb' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'lb', array_call_result_217490)
        
        # Getting the type of 'self' (line 45)
        self_217491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 45)
        lsq_solvers_217492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 26), self_217491, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 45)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 8), lsq_solvers_217492)
        # Getting the type of the for loop variable (line 45)
        for_loop_var_217493 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 8), lsq_solvers_217492)
        # Assigning a type to the variable 'lsq_solver' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'lsq_solver', for_loop_var_217493)
        # SSA begins for a for statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 46):
        
        # Call to lsq_linear(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'A' (line 46)
        A_217495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'A', False)
        # Getting the type of 'b' (line 46)
        b_217496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_217497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'lb' (line 46)
        lb_217498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 36), tuple_217497, lb_217498)
        # Adding element type (line 46)
        # Getting the type of 'np' (line 46)
        np_217499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'np', False)
        # Obtaining the member 'inf' of a type (line 46)
        inf_217500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 40), np_217499, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 36), tuple_217497, inf_217500)
        
        # Processing the call keyword arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_217501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 56), 'self', False)
        # Obtaining the member 'method' of a type (line 46)
        method_217502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 56), self_217501, 'method')
        keyword_217503 = method_217502
        # Getting the type of 'lsq_solver' (line 47)
        lsq_solver_217504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'lsq_solver', False)
        keyword_217505 = lsq_solver_217504
        kwargs_217506 = {'lsq_solver': keyword_217505, 'method': keyword_217503}
        # Getting the type of 'lsq_linear' (line 46)
        lsq_linear_217494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 46)
        lsq_linear_call_result_217507 = invoke(stypy.reporting.localization.Localization(__file__, 46, 18), lsq_linear_217494, *[A_217495, b_217496, tuple_217497], **kwargs_217506)
        
        # Assigning a type to the variable 'res' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'res', lsq_linear_call_result_217507)
        
        # Call to assert_allclose(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'res' (line 48)
        res_217509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 48)
        x_217510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 28), res_217509, 'x')
        
        # Call to array(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_217513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        float_217514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 44), list_217513, float_217514)
        # Adding element type (line 48)
        int_217515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 44), list_217513, int_217515)
        
        # Processing the call keyword arguments (line 48)
        kwargs_217516 = {}
        # Getting the type of 'np' (line 48)
        np_217511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 48)
        array_217512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 35), np_217511, 'array')
        # Calling array(args, kwargs) (line 48)
        array_call_result_217517 = invoke(stypy.reporting.localization.Localization(__file__, 48, 35), array_217512, *[list_217513], **kwargs_217516)
        
        # Processing the call keyword arguments (line 48)
        float_217518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 33), 'float')
        keyword_217519 = float_217518
        kwargs_217520 = {'atol': keyword_217519}
        # Getting the type of 'assert_allclose' (line 48)
        assert_allclose_217508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 48)
        assert_allclose_call_result_217521 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), assert_allclose_217508, *[x_217510, array_call_result_217517], **kwargs_217520)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 51):
        
        # Call to array(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_217524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        # Getting the type of 'np' (line 51)
        np_217525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'np', False)
        # Obtaining the member 'inf' of a type (line 51)
        inf_217526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 23), np_217525, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 22), list_217524, inf_217526)
        # Adding element type (line 51)
        int_217527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 22), list_217524, int_217527)
        
        # Processing the call keyword arguments (line 51)
        kwargs_217528 = {}
        # Getting the type of 'np' (line 51)
        np_217522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 51)
        array_217523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), np_217522, 'array')
        # Calling array(args, kwargs) (line 51)
        array_call_result_217529 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), array_217523, *[list_217524], **kwargs_217528)
        
        # Assigning a type to the variable 'ub' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'ub', array_call_result_217529)
        
        # Getting the type of 'self' (line 52)
        self_217530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 52)
        lsq_solvers_217531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 26), self_217530, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 52)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 8), lsq_solvers_217531)
        # Getting the type of the for loop variable (line 52)
        for_loop_var_217532 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 8), lsq_solvers_217531)
        # Assigning a type to the variable 'lsq_solver' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'lsq_solver', for_loop_var_217532)
        # SSA begins for a for statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 53):
        
        # Call to lsq_linear(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'A' (line 53)
        A_217534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'A', False)
        # Getting the type of 'b' (line 53)
        b_217535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_217536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        
        # Getting the type of 'np' (line 53)
        np_217537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'np', False)
        # Obtaining the member 'inf' of a type (line 53)
        inf_217538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 37), np_217537, 'inf')
        # Applying the 'usub' unary operator (line 53)
        result___neg___217539 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 36), 'usub', inf_217538)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 36), tuple_217536, result___neg___217539)
        # Adding element type (line 53)
        # Getting the type of 'ub' (line 53)
        ub_217540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 45), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 36), tuple_217536, ub_217540)
        
        # Processing the call keyword arguments (line 53)
        # Getting the type of 'self' (line 53)
        self_217541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 57), 'self', False)
        # Obtaining the member 'method' of a type (line 53)
        method_217542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 57), self_217541, 'method')
        keyword_217543 = method_217542
        # Getting the type of 'lsq_solver' (line 54)
        lsq_solver_217544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'lsq_solver', False)
        keyword_217545 = lsq_solver_217544
        kwargs_217546 = {'lsq_solver': keyword_217545, 'method': keyword_217543}
        # Getting the type of 'lsq_linear' (line 53)
        lsq_linear_217533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 53)
        lsq_linear_call_result_217547 = invoke(stypy.reporting.localization.Localization(__file__, 53, 18), lsq_linear_217533, *[A_217534, b_217535, tuple_217536], **kwargs_217546)
        
        # Assigning a type to the variable 'res' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'res', lsq_linear_call_result_217547)
        
        # Call to assert_allclose(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'res' (line 55)
        res_217549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 55)
        x_217550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 28), res_217549, 'x')
        
        # Call to array(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_217553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        float_217554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_217553, float_217554)
        # Adding element type (line 55)
        int_217555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_217553, int_217555)
        
        # Processing the call keyword arguments (line 55)
        kwargs_217556 = {}
        # Getting the type of 'np' (line 55)
        np_217551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 55)
        array_217552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 35), np_217551, 'array')
        # Calling array(args, kwargs) (line 55)
        array_call_result_217557 = invoke(stypy.reporting.localization.Localization(__file__, 55, 35), array_217552, *[list_217553], **kwargs_217556)
        
        # Processing the call keyword arguments (line 55)
        kwargs_217558 = {}
        # Getting the type of 'assert_allclose' (line 55)
        assert_allclose_217548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 55)
        assert_allclose_call_result_217559 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), assert_allclose_217548, *[x_217550, array_call_result_217557], **kwargs_217558)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 57):
        
        # Call to array(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_217562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_217563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 22), list_217562, int_217563)
        # Adding element type (line 57)
        # Getting the type of 'np' (line 57)
        np_217564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'np', False)
        # Obtaining the member 'inf' of a type (line 57)
        inf_217565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 27), np_217564, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 22), list_217562, inf_217565)
        
        # Processing the call keyword arguments (line 57)
        kwargs_217566 = {}
        # Getting the type of 'np' (line 57)
        np_217560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 57)
        array_217561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), np_217560, 'array')
        # Calling array(args, kwargs) (line 57)
        array_call_result_217567 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), array_217561, *[list_217562], **kwargs_217566)
        
        # Assigning a type to the variable 'ub' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'ub', array_call_result_217567)
        
        # Getting the type of 'self' (line 58)
        self_217568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 58)
        lsq_solvers_217569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 26), self_217568, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 58)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 8), lsq_solvers_217569)
        # Getting the type of the for loop variable (line 58)
        for_loop_var_217570 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 8), lsq_solvers_217569)
        # Assigning a type to the variable 'lsq_solver' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'lsq_solver', for_loop_var_217570)
        # SSA begins for a for statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 59):
        
        # Call to lsq_linear(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'A' (line 59)
        A_217572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'A', False)
        # Getting the type of 'b' (line 59)
        b_217573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_217574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        
        # Getting the type of 'np' (line 59)
        np_217575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'np', False)
        # Obtaining the member 'inf' of a type (line 59)
        inf_217576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 37), np_217575, 'inf')
        # Applying the 'usub' unary operator (line 59)
        result___neg___217577 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 36), 'usub', inf_217576)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 36), tuple_217574, result___neg___217577)
        # Adding element type (line 59)
        # Getting the type of 'ub' (line 59)
        ub_217578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 36), tuple_217574, ub_217578)
        
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'self' (line 59)
        self_217579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 57), 'self', False)
        # Obtaining the member 'method' of a type (line 59)
        method_217580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 57), self_217579, 'method')
        keyword_217581 = method_217580
        # Getting the type of 'lsq_solver' (line 60)
        lsq_solver_217582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'lsq_solver', False)
        keyword_217583 = lsq_solver_217582
        kwargs_217584 = {'lsq_solver': keyword_217583, 'method': keyword_217581}
        # Getting the type of 'lsq_linear' (line 59)
        lsq_linear_217571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 59)
        lsq_linear_call_result_217585 = invoke(stypy.reporting.localization.Localization(__file__, 59, 18), lsq_linear_217571, *[A_217572, b_217573, tuple_217574], **kwargs_217584)
        
        # Assigning a type to the variable 'res' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'res', lsq_linear_call_result_217585)
        
        # Call to assert_allclose(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'res' (line 61)
        res_217587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 61)
        x_217588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 28), res_217587, 'x')
        
        # Call to array(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_217591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_217592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 44), list_217591, int_217592)
        # Adding element type (line 61)
        float_217593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 44), list_217591, float_217593)
        
        # Processing the call keyword arguments (line 61)
        kwargs_217594 = {}
        # Getting the type of 'np' (line 61)
        np_217589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 61)
        array_217590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 35), np_217589, 'array')
        # Calling array(args, kwargs) (line 61)
        array_call_result_217595 = invoke(stypy.reporting.localization.Localization(__file__, 61, 35), array_217590, *[list_217591], **kwargs_217594)
        
        # Processing the call keyword arguments (line 61)
        kwargs_217596 = {}
        # Getting the type of 'assert_allclose' (line 61)
        assert_allclose_217586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 61)
        assert_allclose_call_result_217597 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), assert_allclose_217586, *[x_217588, array_call_result_217595], **kwargs_217596)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 63):
        
        # Call to array(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_217600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        int_217601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 22), list_217600, int_217601)
        # Adding element type (line 63)
        int_217602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 22), list_217600, int_217602)
        
        # Processing the call keyword arguments (line 63)
        kwargs_217603 = {}
        # Getting the type of 'np' (line 63)
        np_217598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 63)
        array_217599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 13), np_217598, 'array')
        # Calling array(args, kwargs) (line 63)
        array_call_result_217604 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), array_217599, *[list_217600], **kwargs_217603)
        
        # Assigning a type to the variable 'lb' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'lb', array_call_result_217604)
        
        # Assigning a Call to a Name (line 64):
        
        # Call to array(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_217607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_217608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 22), list_217607, int_217608)
        # Adding element type (line 64)
        int_217609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 22), list_217607, int_217609)
        
        # Processing the call keyword arguments (line 64)
        kwargs_217610 = {}
        # Getting the type of 'np' (line 64)
        np_217605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 64)
        array_217606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 13), np_217605, 'array')
        # Calling array(args, kwargs) (line 64)
        array_call_result_217611 = invoke(stypy.reporting.localization.Localization(__file__, 64, 13), array_217606, *[list_217607], **kwargs_217610)
        
        # Assigning a type to the variable 'ub' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'ub', array_call_result_217611)
        
        # Getting the type of 'self' (line 65)
        self_217612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 65)
        lsq_solvers_217613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), self_217612, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 65)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 8), lsq_solvers_217613)
        # Getting the type of the for loop variable (line 65)
        for_loop_var_217614 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 8), lsq_solvers_217613)
        # Assigning a type to the variable 'lsq_solver' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'lsq_solver', for_loop_var_217614)
        # SSA begins for a for statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 66):
        
        # Call to lsq_linear(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'A' (line 66)
        A_217616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'A', False)
        # Getting the type of 'b' (line 66)
        b_217617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_217618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        # Getting the type of 'lb' (line 66)
        lb_217619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 36), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 36), tuple_217618, lb_217619)
        # Adding element type (line 66)
        # Getting the type of 'ub' (line 66)
        ub_217620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 40), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 36), tuple_217618, ub_217620)
        
        # Processing the call keyword arguments (line 66)
        # Getting the type of 'self' (line 66)
        self_217621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 52), 'self', False)
        # Obtaining the member 'method' of a type (line 66)
        method_217622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 52), self_217621, 'method')
        keyword_217623 = method_217622
        # Getting the type of 'lsq_solver' (line 67)
        lsq_solver_217624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'lsq_solver', False)
        keyword_217625 = lsq_solver_217624
        kwargs_217626 = {'lsq_solver': keyword_217625, 'method': keyword_217623}
        # Getting the type of 'lsq_linear' (line 66)
        lsq_linear_217615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 66)
        lsq_linear_call_result_217627 = invoke(stypy.reporting.localization.Localization(__file__, 66, 18), lsq_linear_217615, *[A_217616, b_217617, tuple_217618], **kwargs_217626)
        
        # Assigning a type to the variable 'res' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'res', lsq_linear_call_result_217627)
        
        # Call to assert_allclose(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'res' (line 68)
        res_217629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 68)
        x_217630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 28), res_217629, 'x')
        
        # Call to array(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_217633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        float_217634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 44), list_217633, float_217634)
        # Adding element type (line 68)
        int_217635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 44), list_217633, int_217635)
        
        # Processing the call keyword arguments (line 68)
        kwargs_217636 = {}
        # Getting the type of 'np' (line 68)
        np_217631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 68)
        array_217632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 35), np_217631, 'array')
        # Calling array(args, kwargs) (line 68)
        array_call_result_217637 = invoke(stypy.reporting.localization.Localization(__file__, 68, 35), array_217632, *[list_217633], **kwargs_217636)
        
        # Processing the call keyword arguments (line 68)
        kwargs_217638 = {}
        # Getting the type of 'assert_allclose' (line 68)
        assert_allclose_217628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 68)
        assert_allclose_call_result_217639 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), assert_allclose_217628, *[x_217630, array_call_result_217637], **kwargs_217638)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dense_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dense_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_217640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217640)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dense_bounds'
        return stypy_return_type_217640


    @norecursion
    def test_dense_rank_deficient(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dense_rank_deficient'
        module_type_store = module_type_store.open_function_context('test_dense_rank_deficient', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_dense_rank_deficient')
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_dense_rank_deficient.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_dense_rank_deficient', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dense_rank_deficient', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dense_rank_deficient(...)' code ##################

        
        # Assigning a Call to a Name (line 71):
        
        # Call to array(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_217643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_217644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        float_217645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_217644, float_217645)
        # Adding element type (line 71)
        float_217646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), list_217644, float_217646)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 21), list_217643, list_217644)
        
        # Processing the call keyword arguments (line 71)
        kwargs_217647 = {}
        # Getting the type of 'np' (line 71)
        np_217641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 71)
        array_217642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), np_217641, 'array')
        # Calling array(args, kwargs) (line 71)
        array_call_result_217648 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), array_217642, *[list_217643], **kwargs_217647)
        
        # Assigning a type to the variable 'A' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'A', array_call_result_217648)
        
        # Assigning a Call to a Name (line 72):
        
        # Call to array(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_217651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        float_217652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), list_217651, float_217652)
        
        # Processing the call keyword arguments (line 72)
        kwargs_217653 = {}
        # Getting the type of 'np' (line 72)
        np_217649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 72)
        array_217650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), np_217649, 'array')
        # Calling array(args, kwargs) (line 72)
        array_call_result_217654 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), array_217650, *[list_217651], **kwargs_217653)
        
        # Assigning a type to the variable 'b' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'b', array_call_result_217654)
        
        # Assigning a List to a Name (line 73):
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_217655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        float_217656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 13), list_217655, float_217656)
        # Adding element type (line 73)
        float_217657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 13), list_217655, float_217657)
        
        # Assigning a type to the variable 'lb' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'lb', list_217655)
        
        # Assigning a List to a Name (line 74):
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_217658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        float_217659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 13), list_217658, float_217659)
        # Adding element type (line 74)
        float_217660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 13), list_217658, float_217660)
        
        # Assigning a type to the variable 'ub' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'ub', list_217658)
        
        # Getting the type of 'self' (line 75)
        self_217661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 75)
        lsq_solvers_217662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 26), self_217661, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 75)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 75, 8), lsq_solvers_217662)
        # Getting the type of the for loop variable (line 75)
        for_loop_var_217663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 75, 8), lsq_solvers_217662)
        # Assigning a type to the variable 'lsq_solver' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'lsq_solver', for_loop_var_217663)
        # SSA begins for a for statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 76):
        
        # Call to lsq_linear(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'A' (line 76)
        A_217665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'A', False)
        # Getting the type of 'b' (line 76)
        b_217666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_217667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'lb' (line 76)
        lb_217668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 36), tuple_217667, lb_217668)
        # Adding element type (line 76)
        # Getting the type of 'ub' (line 76)
        ub_217669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 40), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 36), tuple_217667, ub_217669)
        
        # Processing the call keyword arguments (line 76)
        # Getting the type of 'self' (line 76)
        self_217670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 52), 'self', False)
        # Obtaining the member 'method' of a type (line 76)
        method_217671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 52), self_217670, 'method')
        keyword_217672 = method_217671
        # Getting the type of 'lsq_solver' (line 77)
        lsq_solver_217673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 40), 'lsq_solver', False)
        keyword_217674 = lsq_solver_217673
        kwargs_217675 = {'lsq_solver': keyword_217674, 'method': keyword_217672}
        # Getting the type of 'lsq_linear' (line 76)
        lsq_linear_217664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 76)
        lsq_linear_call_result_217676 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), lsq_linear_217664, *[A_217665, b_217666, tuple_217667], **kwargs_217675)
        
        # Assigning a type to the variable 'res' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'res', lsq_linear_call_result_217676)
        
        # Call to assert_allclose(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'res' (line 78)
        res_217678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 78)
        x_217679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 28), res_217678, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_217680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        float_217681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 35), list_217680, float_217681)
        # Adding element type (line 78)
        float_217682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 35), list_217680, float_217682)
        
        # Processing the call keyword arguments (line 78)
        kwargs_217683 = {}
        # Getting the type of 'assert_allclose' (line 78)
        assert_allclose_217677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 78)
        assert_allclose_call_result_217684 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), assert_allclose_217677, *[x_217679, list_217680], **kwargs_217683)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 80):
        
        # Call to array(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_217687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_217688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        float_217689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 12), list_217688, float_217689)
        # Adding element type (line 81)
        float_217690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 12), list_217688, float_217690)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_217687, list_217688)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_217691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        float_217692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), list_217691, float_217692)
        # Adding element type (line 82)
        float_217693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), list_217691, float_217693)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_217687, list_217691)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_217694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        float_217695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 12), list_217694, float_217695)
        # Adding element type (line 83)
        float_217696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 12), list_217694, float_217696)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_217687, list_217694)
        
        # Processing the call keyword arguments (line 80)
        kwargs_217697 = {}
        # Getting the type of 'np' (line 80)
        np_217685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 80)
        array_217686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), np_217685, 'array')
        # Calling array(args, kwargs) (line 80)
        array_call_result_217698 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), array_217686, *[list_217687], **kwargs_217697)
        
        # Assigning a type to the variable 'A' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'A', array_call_result_217698)
        
        # Assigning a Call to a Name (line 85):
        
        # Call to array(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_217701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        float_217702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_217701, float_217702)
        # Adding element type (line 85)
        float_217703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_217701, float_217703)
        # Adding element type (line 85)
        float_217704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_217701, float_217704)
        
        # Processing the call keyword arguments (line 85)
        kwargs_217705 = {}
        # Getting the type of 'np' (line 85)
        np_217699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 85)
        array_217700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), np_217699, 'array')
        # Calling array(args, kwargs) (line 85)
        array_call_result_217706 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), array_217700, *[list_217701], **kwargs_217705)
        
        # Assigning a type to the variable 'b' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'b', array_call_result_217706)
        
        # Assigning a List to a Name (line 86):
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_217707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        int_217708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 13), list_217707, int_217708)
        # Adding element type (line 86)
        int_217709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 13), list_217707, int_217709)
        
        # Assigning a type to the variable 'lb' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'lb', list_217707)
        
        # Assigning a List to a Name (line 87):
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_217710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        int_217711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 13), list_217710, int_217711)
        # Adding element type (line 87)
        float_217712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 13), list_217710, float_217712)
        
        # Assigning a type to the variable 'ub' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'ub', list_217710)
        
        # Getting the type of 'self' (line 88)
        self_217713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'self')
        # Obtaining the member 'lsq_solvers' of a type (line 88)
        lsq_solvers_217714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 26), self_217713, 'lsq_solvers')
        # Testing the type of a for loop iterable (line 88)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), lsq_solvers_217714)
        # Getting the type of the for loop variable (line 88)
        for_loop_var_217715 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), lsq_solvers_217714)
        # Assigning a type to the variable 'lsq_solver' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'lsq_solver', for_loop_var_217715)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 89):
        
        # Call to lsq_linear(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'A' (line 89)
        A_217717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'A', False)
        # Getting the type of 'b' (line 89)
        b_217718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 32), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 89)
        tuple_217719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 89)
        # Adding element type (line 89)
        # Getting the type of 'lb' (line 89)
        lb_217720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 36), tuple_217719, lb_217720)
        # Adding element type (line 89)
        # Getting the type of 'ub' (line 89)
        ub_217721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 36), tuple_217719, ub_217721)
        
        # Processing the call keyword arguments (line 89)
        # Getting the type of 'self' (line 89)
        self_217722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 52), 'self', False)
        # Obtaining the member 'method' of a type (line 89)
        method_217723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 52), self_217722, 'method')
        keyword_217724 = method_217723
        # Getting the type of 'lsq_solver' (line 90)
        lsq_solver_217725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'lsq_solver', False)
        keyword_217726 = lsq_solver_217725
        kwargs_217727 = {'lsq_solver': keyword_217726, 'method': keyword_217724}
        # Getting the type of 'lsq_linear' (line 89)
        lsq_linear_217716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 89)
        lsq_linear_call_result_217728 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), lsq_linear_217716, *[A_217717, b_217718, tuple_217719], **kwargs_217727)
        
        # Assigning a type to the variable 'res' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'res', lsq_linear_call_result_217728)
        
        # Call to assert_allclose(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'res' (line 91)
        res_217730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'res', False)
        # Obtaining the member 'optimality' of a type (line 91)
        optimality_217731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 28), res_217730, 'optimality')
        int_217732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 44), 'int')
        # Processing the call keyword arguments (line 91)
        float_217733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 52), 'float')
        keyword_217734 = float_217733
        kwargs_217735 = {'atol': keyword_217734}
        # Getting the type of 'assert_allclose' (line 91)
        assert_allclose_217729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 91)
        assert_allclose_call_result_217736 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), assert_allclose_217729, *[optimality_217731, int_217732], **kwargs_217735)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dense_rank_deficient(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dense_rank_deficient' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_217737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217737)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dense_rank_deficient'
        return stypy_return_type_217737


    @norecursion
    def test_full_result(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_full_result'
        module_type_store = module_type_store.open_function_context('test_full_result', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_full_result')
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_full_result', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_full_result', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_full_result(...)' code ##################

        
        # Assigning a Call to a Name (line 94):
        
        # Call to array(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_217740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        int_217741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 22), list_217740, int_217741)
        # Adding element type (line 94)
        int_217742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 22), list_217740, int_217742)
        
        # Processing the call keyword arguments (line 94)
        kwargs_217743 = {}
        # Getting the type of 'np' (line 94)
        np_217738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 94)
        array_217739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 13), np_217738, 'array')
        # Calling array(args, kwargs) (line 94)
        array_call_result_217744 = invoke(stypy.reporting.localization.Localization(__file__, 94, 13), array_217739, *[list_217740], **kwargs_217743)
        
        # Assigning a type to the variable 'lb' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'lb', array_call_result_217744)
        
        # Assigning a Call to a Name (line 95):
        
        # Call to array(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_217747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        int_217748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 22), list_217747, int_217748)
        # Adding element type (line 95)
        int_217749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 22), list_217747, int_217749)
        
        # Processing the call keyword arguments (line 95)
        kwargs_217750 = {}
        # Getting the type of 'np' (line 95)
        np_217745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 95)
        array_217746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 13), np_217745, 'array')
        # Calling array(args, kwargs) (line 95)
        array_call_result_217751 = invoke(stypy.reporting.localization.Localization(__file__, 95, 13), array_217746, *[list_217747], **kwargs_217750)
        
        # Assigning a type to the variable 'ub' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'ub', array_call_result_217751)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to lsq_linear(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'A' (line 96)
        A_217753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'A', False)
        # Getting the type of 'b' (line 96)
        b_217754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_217755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        # Getting the type of 'lb' (line 96)
        lb_217756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 32), tuple_217755, lb_217756)
        # Adding element type (line 96)
        # Getting the type of 'ub' (line 96)
        ub_217757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 36), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 32), tuple_217755, ub_217757)
        
        # Processing the call keyword arguments (line 96)
        # Getting the type of 'self' (line 96)
        self_217758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 48), 'self', False)
        # Obtaining the member 'method' of a type (line 96)
        method_217759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 48), self_217758, 'method')
        keyword_217760 = method_217759
        kwargs_217761 = {'method': keyword_217760}
        # Getting the type of 'lsq_linear' (line 96)
        lsq_linear_217752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 96)
        lsq_linear_call_result_217762 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), lsq_linear_217752, *[A_217753, b_217754, tuple_217755], **kwargs_217761)
        
        # Assigning a type to the variable 'res' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'res', lsq_linear_call_result_217762)
        
        # Call to assert_allclose(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'res' (line 98)
        res_217764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 98)
        x_217765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), res_217764, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_217766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        float_217767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 31), list_217766, float_217767)
        # Adding element type (line 98)
        int_217768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 31), list_217766, int_217768)
        
        # Processing the call keyword arguments (line 98)
        kwargs_217769 = {}
        # Getting the type of 'assert_allclose' (line 98)
        assert_allclose_217763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 98)
        assert_allclose_call_result_217770 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), assert_allclose_217763, *[x_217765, list_217766], **kwargs_217769)
        
        
        # Assigning a BinOp to a Name (line 100):
        
        # Call to dot(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'res' (line 100)
        res_217773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'res', False)
        # Obtaining the member 'x' of a type (line 100)
        x_217774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 18), res_217773, 'x')
        # Processing the call keyword arguments (line 100)
        kwargs_217775 = {}
        # Getting the type of 'A' (line 100)
        A_217771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'A', False)
        # Obtaining the member 'dot' of a type (line 100)
        dot_217772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), A_217771, 'dot')
        # Calling dot(args, kwargs) (line 100)
        dot_call_result_217776 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), dot_217772, *[x_217774], **kwargs_217775)
        
        # Getting the type of 'b' (line 100)
        b_217777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'b')
        # Applying the binary operator '-' (line 100)
        result_sub_217778 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 12), '-', dot_call_result_217776, b_217777)
        
        # Assigning a type to the variable 'r' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'r', result_sub_217778)
        
        # Call to assert_allclose(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'res' (line 101)
        res_217780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'res', False)
        # Obtaining the member 'cost' of a type (line 101)
        cost_217781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 24), res_217780, 'cost')
        float_217782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 34), 'float')
        
        # Call to dot(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'r' (line 101)
        r_217785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 47), 'r', False)
        # Getting the type of 'r' (line 101)
        r_217786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'r', False)
        # Processing the call keyword arguments (line 101)
        kwargs_217787 = {}
        # Getting the type of 'np' (line 101)
        np_217783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'np', False)
        # Obtaining the member 'dot' of a type (line 101)
        dot_217784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 40), np_217783, 'dot')
        # Calling dot(args, kwargs) (line 101)
        dot_call_result_217788 = invoke(stypy.reporting.localization.Localization(__file__, 101, 40), dot_217784, *[r_217785, r_217786], **kwargs_217787)
        
        # Applying the binary operator '*' (line 101)
        result_mul_217789 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 34), '*', float_217782, dot_call_result_217788)
        
        # Processing the call keyword arguments (line 101)
        kwargs_217790 = {}
        # Getting the type of 'assert_allclose' (line 101)
        assert_allclose_217779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 101)
        assert_allclose_call_result_217791 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assert_allclose_217779, *[cost_217781, result_mul_217789], **kwargs_217790)
        
        
        # Call to assert_allclose(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'res' (line 102)
        res_217793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'res', False)
        # Obtaining the member 'fun' of a type (line 102)
        fun_217794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), res_217793, 'fun')
        # Getting the type of 'r' (line 102)
        r_217795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 33), 'r', False)
        # Processing the call keyword arguments (line 102)
        kwargs_217796 = {}
        # Getting the type of 'assert_allclose' (line 102)
        assert_allclose_217792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 102)
        assert_allclose_call_result_217797 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_allclose_217792, *[fun_217794, r_217795], **kwargs_217796)
        
        
        # Call to assert_allclose(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'res' (line 104)
        res_217799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'res', False)
        # Obtaining the member 'optimality' of a type (line 104)
        optimality_217800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 24), res_217799, 'optimality')
        float_217801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'float')
        # Processing the call keyword arguments (line 104)
        float_217802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 50), 'float')
        keyword_217803 = float_217802
        kwargs_217804 = {'atol': keyword_217803}
        # Getting the type of 'assert_allclose' (line 104)
        assert_allclose_217798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 104)
        assert_allclose_call_result_217805 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assert_allclose_217798, *[optimality_217800, float_217801], **kwargs_217804)
        
        
        # Call to assert_equal(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'res' (line 105)
        res_217807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'res', False)
        # Obtaining the member 'active_mask' of a type (line 105)
        active_mask_217808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 21), res_217807, 'active_mask')
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_217809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        int_217810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 38), list_217809, int_217810)
        # Adding element type (line 105)
        int_217811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 38), list_217809, int_217811)
        
        # Processing the call keyword arguments (line 105)
        kwargs_217812 = {}
        # Getting the type of 'assert_equal' (line 105)
        assert_equal_217806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 105)
        assert_equal_call_result_217813 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assert_equal_217806, *[active_mask_217808, list_217809], **kwargs_217812)
        
        
        # Call to assert_(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Getting the type of 'res' (line 106)
        res_217815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'res', False)
        # Obtaining the member 'nit' of a type (line 106)
        nit_217816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), res_217815, 'nit')
        int_217817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 26), 'int')
        # Applying the binary operator '<' (line 106)
        result_lt_217818 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 16), '<', nit_217816, int_217817)
        
        # Processing the call keyword arguments (line 106)
        kwargs_217819 = {}
        # Getting the type of 'assert_' (line 106)
        assert__217814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 106)
        assert__call_result_217820 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert__217814, *[result_lt_217818], **kwargs_217819)
        
        
        # Call to assert_(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'res' (line 107)
        res_217822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'res', False)
        # Obtaining the member 'status' of a type (line 107)
        status_217823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), res_217822, 'status')
        int_217824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'int')
        # Applying the binary operator '==' (line 107)
        result_eq_217825 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 16), '==', status_217823, int_217824)
        
        
        # Getting the type of 'res' (line 107)
        res_217826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 35), 'res', False)
        # Obtaining the member 'status' of a type (line 107)
        status_217827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 35), res_217826, 'status')
        int_217828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 49), 'int')
        # Applying the binary operator '==' (line 107)
        result_eq_217829 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 35), '==', status_217827, int_217828)
        
        # Applying the binary operator 'or' (line 107)
        result_or_keyword_217830 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 16), 'or', result_eq_217825, result_eq_217829)
        
        # Processing the call keyword arguments (line 107)
        kwargs_217831 = {}
        # Getting the type of 'assert_' (line 107)
        assert__217821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 107)
        assert__call_result_217832 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert__217821, *[result_or_keyword_217830], **kwargs_217831)
        
        
        # Call to assert_(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to isinstance(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'res' (line 108)
        res_217835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'res', False)
        # Obtaining the member 'message' of a type (line 108)
        message_217836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), res_217835, 'message')
        # Getting the type of 'str' (line 108)
        str_217837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'str', False)
        # Processing the call keyword arguments (line 108)
        kwargs_217838 = {}
        # Getting the type of 'isinstance' (line 108)
        isinstance_217834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 108)
        isinstance_call_result_217839 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), isinstance_217834, *[message_217836, str_217837], **kwargs_217838)
        
        # Processing the call keyword arguments (line 108)
        kwargs_217840 = {}
        # Getting the type of 'assert_' (line 108)
        assert__217833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 108)
        assert__call_result_217841 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assert__217833, *[isinstance_call_result_217839], **kwargs_217840)
        
        
        # Call to assert_(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'res' (line 109)
        res_217843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'res', False)
        # Obtaining the member 'success' of a type (line 109)
        success_217844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), res_217843, 'success')
        # Processing the call keyword arguments (line 109)
        kwargs_217845 = {}
        # Getting the type of 'assert_' (line 109)
        assert__217842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 109)
        assert__call_result_217846 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), assert__217842, *[success_217844], **kwargs_217845)
        
        
        # ################# End of 'test_full_result(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_full_result' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_217847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_full_result'
        return stypy_return_type_217847


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseMixin' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'BaseMixin', BaseMixin)
# Declaration of the 'SparseMixin' class

class SparseMixin(object, ):

    @norecursion
    def test_sparse_and_LinearOperator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_and_LinearOperator'
        module_type_store = module_type_store.open_function_context('test_sparse_and_LinearOperator', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_sparse_and_LinearOperator')
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_sparse_and_LinearOperator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_sparse_and_LinearOperator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_and_LinearOperator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_and_LinearOperator(...)' code ##################

        
        # Assigning a Num to a Name (line 114):
        int_217848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
        # Assigning a type to the variable 'm' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'm', int_217848)
        
        # Assigning a Num to a Name (line 115):
        int_217849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
        # Assigning a type to the variable 'n' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'n', int_217849)
        
        # Assigning a Call to a Name (line 116):
        
        # Call to rand(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'm' (line 116)
        m_217851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'm', False)
        # Getting the type of 'n' (line 116)
        n_217852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'n', False)
        # Processing the call keyword arguments (line 116)
        int_217853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 36), 'int')
        keyword_217854 = int_217853
        kwargs_217855 = {'random_state': keyword_217854}
        # Getting the type of 'rand' (line 116)
        rand_217850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'rand', False)
        # Calling rand(args, kwargs) (line 116)
        rand_call_result_217856 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), rand_217850, *[m_217851, n_217852], **kwargs_217855)
        
        # Assigning a type to the variable 'A' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'A', rand_call_result_217856)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to randn(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'm' (line 117)
        m_217860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'm', False)
        # Processing the call keyword arguments (line 117)
        kwargs_217861 = {}
        # Getting the type of 'self' (line 117)
        self_217857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'self', False)
        # Obtaining the member 'rnd' of a type (line 117)
        rnd_217858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), self_217857, 'rnd')
        # Obtaining the member 'randn' of a type (line 117)
        randn_217859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), rnd_217858, 'randn')
        # Calling randn(args, kwargs) (line 117)
        randn_call_result_217862 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), randn_217859, *[m_217860], **kwargs_217861)
        
        # Assigning a type to the variable 'b' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'b', randn_call_result_217862)
        
        # Assigning a Call to a Name (line 118):
        
        # Call to lsq_linear(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'A' (line 118)
        A_217864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'A', False)
        # Getting the type of 'b' (line 118)
        b_217865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'b', False)
        # Processing the call keyword arguments (line 118)
        kwargs_217866 = {}
        # Getting the type of 'lsq_linear' (line 118)
        lsq_linear_217863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 118)
        lsq_linear_call_result_217867 = invoke(stypy.reporting.localization.Localization(__file__, 118, 14), lsq_linear_217863, *[A_217864, b_217865], **kwargs_217866)
        
        # Assigning a type to the variable 'res' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'res', lsq_linear_call_result_217867)
        
        # Call to assert_allclose(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'res' (line 119)
        res_217869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'res', False)
        # Obtaining the member 'optimality' of a type (line 119)
        optimality_217870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 24), res_217869, 'optimality')
        int_217871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 40), 'int')
        # Processing the call keyword arguments (line 119)
        float_217872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 48), 'float')
        keyword_217873 = float_217872
        kwargs_217874 = {'atol': keyword_217873}
        # Getting the type of 'assert_allclose' (line 119)
        assert_allclose_217868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 119)
        assert_allclose_call_result_217875 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), assert_allclose_217868, *[optimality_217870, int_217871], **kwargs_217874)
        
        
        # Assigning a Call to a Name (line 121):
        
        # Call to aslinearoperator(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'A' (line 121)
        A_217877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'A', False)
        # Processing the call keyword arguments (line 121)
        kwargs_217878 = {}
        # Getting the type of 'aslinearoperator' (line 121)
        aslinearoperator_217876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'aslinearoperator', False)
        # Calling aslinearoperator(args, kwargs) (line 121)
        aslinearoperator_call_result_217879 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), aslinearoperator_217876, *[A_217877], **kwargs_217878)
        
        # Assigning a type to the variable 'A' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'A', aslinearoperator_call_result_217879)
        
        # Assigning a Call to a Name (line 122):
        
        # Call to lsq_linear(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'A' (line 122)
        A_217881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'A', False)
        # Getting the type of 'b' (line 122)
        b_217882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'b', False)
        # Processing the call keyword arguments (line 122)
        kwargs_217883 = {}
        # Getting the type of 'lsq_linear' (line 122)
        lsq_linear_217880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 122)
        lsq_linear_call_result_217884 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), lsq_linear_217880, *[A_217881, b_217882], **kwargs_217883)
        
        # Assigning a type to the variable 'res' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'res', lsq_linear_call_result_217884)
        
        # Call to assert_allclose(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'res' (line 123)
        res_217886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'res', False)
        # Obtaining the member 'optimality' of a type (line 123)
        optimality_217887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 24), res_217886, 'optimality')
        int_217888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 40), 'int')
        # Processing the call keyword arguments (line 123)
        float_217889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 48), 'float')
        keyword_217890 = float_217889
        kwargs_217891 = {'atol': keyword_217890}
        # Getting the type of 'assert_allclose' (line 123)
        assert_allclose_217885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 123)
        assert_allclose_call_result_217892 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), assert_allclose_217885, *[optimality_217887, int_217888], **kwargs_217891)
        
        
        # ################# End of 'test_sparse_and_LinearOperator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_and_LinearOperator' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_217893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217893)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_and_LinearOperator'
        return stypy_return_type_217893


    @norecursion
    def test_sparse_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_bounds'
        module_type_store = module_type_store.open_function_context('test_sparse_bounds', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_sparse_bounds')
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_sparse_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_sparse_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_bounds(...)' code ##################

        
        # Assigning a Num to a Name (line 126):
        int_217894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 12), 'int')
        # Assigning a type to the variable 'm' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'm', int_217894)
        
        # Assigning a Num to a Name (line 127):
        int_217895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
        # Assigning a type to the variable 'n' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'n', int_217895)
        
        # Assigning a Call to a Name (line 128):
        
        # Call to rand(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'm' (line 128)
        m_217897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'm', False)
        # Getting the type of 'n' (line 128)
        n_217898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'n', False)
        # Processing the call keyword arguments (line 128)
        int_217899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'int')
        keyword_217900 = int_217899
        kwargs_217901 = {'random_state': keyword_217900}
        # Getting the type of 'rand' (line 128)
        rand_217896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'rand', False)
        # Calling rand(args, kwargs) (line 128)
        rand_call_result_217902 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), rand_217896, *[m_217897, n_217898], **kwargs_217901)
        
        # Assigning a type to the variable 'A' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'A', rand_call_result_217902)
        
        # Assigning a Call to a Name (line 129):
        
        # Call to randn(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'm' (line 129)
        m_217906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'm', False)
        # Processing the call keyword arguments (line 129)
        kwargs_217907 = {}
        # Getting the type of 'self' (line 129)
        self_217903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'self', False)
        # Obtaining the member 'rnd' of a type (line 129)
        rnd_217904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), self_217903, 'rnd')
        # Obtaining the member 'randn' of a type (line 129)
        randn_217905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), rnd_217904, 'randn')
        # Calling randn(args, kwargs) (line 129)
        randn_call_result_217908 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), randn_217905, *[m_217906], **kwargs_217907)
        
        # Assigning a type to the variable 'b' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'b', randn_call_result_217908)
        
        # Assigning a Call to a Name (line 130):
        
        # Call to randn(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'n' (line 130)
        n_217912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'n', False)
        # Processing the call keyword arguments (line 130)
        kwargs_217913 = {}
        # Getting the type of 'self' (line 130)
        self_217909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'self', False)
        # Obtaining the member 'rnd' of a type (line 130)
        rnd_217910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 13), self_217909, 'rnd')
        # Obtaining the member 'randn' of a type (line 130)
        randn_217911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 13), rnd_217910, 'randn')
        # Calling randn(args, kwargs) (line 130)
        randn_call_result_217914 = invoke(stypy.reporting.localization.Localization(__file__, 130, 13), randn_217911, *[n_217912], **kwargs_217913)
        
        # Assigning a type to the variable 'lb' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'lb', randn_call_result_217914)
        
        # Assigning a BinOp to a Name (line 131):
        # Getting the type of 'lb' (line 131)
        lb_217915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'lb')
        int_217916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 18), 'int')
        # Applying the binary operator '+' (line 131)
        result_add_217917 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 13), '+', lb_217915, int_217916)
        
        # Assigning a type to the variable 'ub' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'ub', result_add_217917)
        
        # Assigning a Call to a Name (line 132):
        
        # Call to lsq_linear(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'A' (line 132)
        A_217919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'A', False)
        # Getting the type of 'b' (line 132)
        b_217920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 132)
        tuple_217921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 132)
        # Adding element type (line 132)
        # Getting the type of 'lb' (line 132)
        lb_217922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 32), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 32), tuple_217921, lb_217922)
        # Adding element type (line 132)
        # Getting the type of 'ub' (line 132)
        ub_217923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 32), tuple_217921, ub_217923)
        
        # Processing the call keyword arguments (line 132)
        kwargs_217924 = {}
        # Getting the type of 'lsq_linear' (line 132)
        lsq_linear_217918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 132)
        lsq_linear_call_result_217925 = invoke(stypy.reporting.localization.Localization(__file__, 132, 14), lsq_linear_217918, *[A_217919, b_217920, tuple_217921], **kwargs_217924)
        
        # Assigning a type to the variable 'res' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'res', lsq_linear_call_result_217925)
        
        # Call to assert_allclose(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'res' (line 133)
        res_217927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'res', False)
        # Obtaining the member 'optimality' of a type (line 133)
        optimality_217928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), res_217927, 'optimality')
        float_217929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 40), 'float')
        # Processing the call keyword arguments (line 133)
        float_217930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 50), 'float')
        keyword_217931 = float_217930
        kwargs_217932 = {'atol': keyword_217931}
        # Getting the type of 'assert_allclose' (line 133)
        assert_allclose_217926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 133)
        assert_allclose_call_result_217933 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assert_allclose_217926, *[optimality_217928, float_217929], **kwargs_217932)
        
        
        # Assigning a Call to a Name (line 135):
        
        # Call to lsq_linear(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'A' (line 135)
        A_217935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'A', False)
        # Getting the type of 'b' (line 135)
        b_217936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_217937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        # Getting the type of 'lb' (line 135)
        lb_217938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 32), tuple_217937, lb_217938)
        # Adding element type (line 135)
        # Getting the type of 'ub' (line 135)
        ub_217939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 32), tuple_217937, ub_217939)
        
        # Processing the call keyword arguments (line 135)
        float_217940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 50), 'float')
        keyword_217941 = float_217940
        kwargs_217942 = {'lsmr_tol': keyword_217941}
        # Getting the type of 'lsq_linear' (line 135)
        lsq_linear_217934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 135)
        lsq_linear_call_result_217943 = invoke(stypy.reporting.localization.Localization(__file__, 135, 14), lsq_linear_217934, *[A_217935, b_217936, tuple_217937], **kwargs_217942)
        
        # Assigning a type to the variable 'res' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'res', lsq_linear_call_result_217943)
        
        # Call to assert_allclose(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'res' (line 136)
        res_217945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'res', False)
        # Obtaining the member 'optimality' of a type (line 136)
        optimality_217946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 24), res_217945, 'optimality')
        float_217947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 40), 'float')
        # Processing the call keyword arguments (line 136)
        float_217948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 50), 'float')
        keyword_217949 = float_217948
        kwargs_217950 = {'atol': keyword_217949}
        # Getting the type of 'assert_allclose' (line 136)
        assert_allclose_217944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 136)
        assert_allclose_call_result_217951 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), assert_allclose_217944, *[optimality_217946, float_217947], **kwargs_217950)
        
        
        # Assigning a Call to a Name (line 138):
        
        # Call to lsq_linear(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'A' (line 138)
        A_217953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'A', False)
        # Getting the type of 'b' (line 138)
        b_217954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'b', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 138)
        tuple_217955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 138)
        # Adding element type (line 138)
        # Getting the type of 'lb' (line 138)
        lb_217956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 32), tuple_217955, lb_217956)
        # Adding element type (line 138)
        # Getting the type of 'ub' (line 138)
        ub_217957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 36), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 32), tuple_217955, ub_217957)
        
        # Processing the call keyword arguments (line 138)
        str_217958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 50), 'str', 'auto')
        keyword_217959 = str_217958
        kwargs_217960 = {'lsmr_tol': keyword_217959}
        # Getting the type of 'lsq_linear' (line 138)
        lsq_linear_217952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 14), 'lsq_linear', False)
        # Calling lsq_linear(args, kwargs) (line 138)
        lsq_linear_call_result_217961 = invoke(stypy.reporting.localization.Localization(__file__, 138, 14), lsq_linear_217952, *[A_217953, b_217954, tuple_217955], **kwargs_217960)
        
        # Assigning a type to the variable 'res' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'res', lsq_linear_call_result_217961)
        
        # Call to assert_allclose(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'res' (line 139)
        res_217963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'res', False)
        # Obtaining the member 'optimality' of a type (line 139)
        optimality_217964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 24), res_217963, 'optimality')
        float_217965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 40), 'float')
        # Processing the call keyword arguments (line 139)
        float_217966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 50), 'float')
        keyword_217967 = float_217966
        kwargs_217968 = {'atol': keyword_217967}
        # Getting the type of 'assert_allclose' (line 139)
        assert_allclose_217962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 139)
        assert_allclose_call_result_217969 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assert_allclose_217962, *[optimality_217964, float_217965], **kwargs_217968)
        
        
        # ################# End of 'test_sparse_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_217970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217970)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_bounds'
        return stypy_return_type_217970


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 112, 0, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SparseMixin' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'SparseMixin', SparseMixin)
# Declaration of the 'TestTRF' class
# Getting the type of 'BaseMixin' (line 142)
BaseMixin_217971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'BaseMixin')
# Getting the type of 'SparseMixin' (line 142)
SparseMixin_217972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'SparseMixin')

class TestTRF(BaseMixin_217971, SparseMixin_217972, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 142, 0, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTRF.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTRF' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'TestTRF', TestTRF)

# Assigning a Str to a Name (line 143):
str_217973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 13), 'str', 'trf')
# Getting the type of 'TestTRF'
TestTRF_217974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestTRF')
# Setting the type of the member 'method' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestTRF_217974, 'method', str_217973)

# Assigning a List to a Name (line 144):

# Obtaining an instance of the builtin type 'list' (line 144)
list_217975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 144)
# Adding element type (line 144)
str_217976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 19), 'str', 'exact')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 18), list_217975, str_217976)
# Adding element type (line 144)
str_217977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 28), 'str', 'lsmr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 18), list_217975, str_217977)

# Getting the type of 'TestTRF'
TestTRF_217978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestTRF')
# Setting the type of the member 'lsq_solvers' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestTRF_217978, 'lsq_solvers', list_217975)
# Declaration of the 'TestBVLS' class
# Getting the type of 'BaseMixin' (line 147)
BaseMixin_217979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'BaseMixin')

class TestBVLS(BaseMixin_217979, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 147, 0, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBVLS.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBVLS' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'TestBVLS', TestBVLS)

# Assigning a Str to a Name (line 148):
str_217980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 13), 'str', 'bvls')
# Getting the type of 'TestBVLS'
TestBVLS_217981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestBVLS')
# Setting the type of the member 'method' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestBVLS_217981, 'method', str_217980)

# Assigning a List to a Name (line 149):

# Obtaining an instance of the builtin type 'list' (line 149)
list_217982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 149)
# Adding element type (line 149)
str_217983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 19), 'str', 'exact')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 18), list_217982, str_217983)

# Getting the type of 'TestBVLS'
TestBVLS_217984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestBVLS')
# Setting the type of the member 'lsq_solvers' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestBVLS_217984, 'lsq_solvers', list_217982)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
