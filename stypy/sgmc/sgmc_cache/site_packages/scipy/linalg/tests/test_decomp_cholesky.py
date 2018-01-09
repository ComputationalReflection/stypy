
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy.testing import assert_array_almost_equal, assert_array_equal
4: from pytest import raises as assert_raises
5: 
6: from numpy import array, transpose, dot, conjugate, zeros_like, empty
7: from numpy.random import random
8: from scipy.linalg import cholesky, cholesky_banded, cho_solve_banded, \
9:      cho_factor, cho_solve
10: 
11: from scipy.linalg._testutils import assert_no_overwrite
12: 
13: 
14: class TestCholesky(object):
15: 
16:     def test_simple(self):
17:         a = [[8, 2, 3], [2, 9, 3], [3, 3, 6]]
18:         c = cholesky(a)
19:         assert_array_almost_equal(dot(transpose(c), c), a)
20:         c = transpose(c)
21:         a = dot(c, transpose(c))
22:         assert_array_almost_equal(cholesky(a, lower=1), c)
23: 
24:     def test_check_finite(self):
25:         a = [[8, 2, 3], [2, 9, 3], [3, 3, 6]]
26:         c = cholesky(a, check_finite=False)
27:         assert_array_almost_equal(dot(transpose(c), c), a)
28:         c = transpose(c)
29:         a = dot(c, transpose(c))
30:         assert_array_almost_equal(cholesky(a, lower=1, check_finite=False), c)
31: 
32:     def test_simple_complex(self):
33:         m = array([[3+1j, 3+4j, 5], [0, 2+2j, 2+7j], [0, 0, 7+4j]])
34:         a = dot(transpose(conjugate(m)), m)
35:         c = cholesky(a)
36:         a1 = dot(transpose(conjugate(c)), c)
37:         assert_array_almost_equal(a, a1)
38:         c = transpose(c)
39:         a = dot(c, transpose(conjugate(c)))
40:         assert_array_almost_equal(cholesky(a, lower=1), c)
41: 
42:     def test_random(self):
43:         n = 20
44:         for k in range(2):
45:             m = random([n, n])
46:             for i in range(n):
47:                 m[i, i] = 20*(.1+m[i, i])
48:             a = dot(transpose(m), m)
49:             c = cholesky(a)
50:             a1 = dot(transpose(c), c)
51:             assert_array_almost_equal(a, a1)
52:             c = transpose(c)
53:             a = dot(c, transpose(c))
54:             assert_array_almost_equal(cholesky(a, lower=1), c)
55: 
56:     def test_random_complex(self):
57:         n = 20
58:         for k in range(2):
59:             m = random([n, n])+1j*random([n, n])
60:             for i in range(n):
61:                 m[i, i] = 20*(.1+abs(m[i, i]))
62:             a = dot(transpose(conjugate(m)), m)
63:             c = cholesky(a)
64:             a1 = dot(transpose(conjugate(c)), c)
65:             assert_array_almost_equal(a, a1)
66:             c = transpose(c)
67:             a = dot(c, transpose(conjugate(c)))
68:             assert_array_almost_equal(cholesky(a, lower=1), c)
69: 
70: 
71: class TestCholeskyBanded(object):
72:     '''Tests for cholesky_banded() and cho_solve_banded.'''
73: 
74:     def test_check_finite(self):
75:         # Symmetric positive definite banded matrix `a`
76:         a = array([[4.0, 1.0, 0.0, 0.0],
77:                    [1.0, 4.0, 0.5, 0.0],
78:                    [0.0, 0.5, 4.0, 0.2],
79:                    [0.0, 0.0, 0.2, 4.0]])
80:         # Banded storage form of `a`.
81:         ab = array([[-1.0, 1.0, 0.5, 0.2],
82:                     [4.0, 4.0, 4.0, 4.0]])
83:         c = cholesky_banded(ab, lower=False, check_finite=False)
84:         ufac = zeros_like(a)
85:         ufac[list(range(4)), list(range(4))] = c[-1]
86:         ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
87:         assert_array_almost_equal(a, dot(ufac.T, ufac))
88: 
89:         b = array([0.0, 0.5, 4.2, 4.2])
90:         x = cho_solve_banded((c, False), b, check_finite=False)
91:         assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])
92: 
93:     def test_upper_real(self):
94:         # Symmetric positive definite banded matrix `a`
95:         a = array([[4.0, 1.0, 0.0, 0.0],
96:                    [1.0, 4.0, 0.5, 0.0],
97:                    [0.0, 0.5, 4.0, 0.2],
98:                    [0.0, 0.0, 0.2, 4.0]])
99:         # Banded storage form of `a`.
100:         ab = array([[-1.0, 1.0, 0.5, 0.2],
101:                     [4.0, 4.0, 4.0, 4.0]])
102:         c = cholesky_banded(ab, lower=False)
103:         ufac = zeros_like(a)
104:         ufac[list(range(4)), list(range(4))] = c[-1]
105:         ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
106:         assert_array_almost_equal(a, dot(ufac.T, ufac))
107: 
108:         b = array([0.0, 0.5, 4.2, 4.2])
109:         x = cho_solve_banded((c, False), b)
110:         assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])
111: 
112:     def test_upper_complex(self):
113:         # Hermitian positive definite banded matrix `a`
114:         a = array([[4.0, 1.0, 0.0, 0.0],
115:                    [1.0, 4.0, 0.5, 0.0],
116:                    [0.0, 0.5, 4.0, -0.2j],
117:                    [0.0, 0.0, 0.2j, 4.0]])
118:         # Banded storage form of `a`.
119:         ab = array([[-1.0, 1.0, 0.5, -0.2j],
120:                     [4.0, 4.0, 4.0, 4.0]])
121:         c = cholesky_banded(ab, lower=False)
122:         ufac = zeros_like(a)
123:         ufac[list(range(4)), list(range(4))] = c[-1]
124:         ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
125:         assert_array_almost_equal(a, dot(ufac.conj().T, ufac))
126: 
127:         b = array([0.0, 0.5, 4.0-0.2j, 0.2j + 4.0])
128:         x = cho_solve_banded((c, False), b)
129:         assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])
130: 
131:     def test_lower_real(self):
132:         # Symmetric positive definite banded matrix `a`
133:         a = array([[4.0, 1.0, 0.0, 0.0],
134:                    [1.0, 4.0, 0.5, 0.0],
135:                    [0.0, 0.5, 4.0, 0.2],
136:                    [0.0, 0.0, 0.2, 4.0]])
137:         # Banded storage form of `a`.
138:         ab = array([[4.0, 4.0, 4.0, 4.0],
139:                     [1.0, 0.5, 0.2, -1.0]])
140:         c = cholesky_banded(ab, lower=True)
141:         lfac = zeros_like(a)
142:         lfac[list(range(4)), list(range(4))] = c[0]
143:         lfac[(1, 2, 3), (0, 1, 2)] = c[1, :3]
144:         assert_array_almost_equal(a, dot(lfac, lfac.T))
145: 
146:         b = array([0.0, 0.5, 4.2, 4.2])
147:         x = cho_solve_banded((c, True), b)
148:         assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])
149: 
150:     def test_lower_complex(self):
151:         # Hermitian positive definite banded matrix `a`
152:         a = array([[4.0, 1.0, 0.0, 0.0],
153:                    [1.0, 4.0, 0.5, 0.0],
154:                    [0.0, 0.5, 4.0, -0.2j],
155:                    [0.0, 0.0, 0.2j, 4.0]])
156:         # Banded storage form of `a`.
157:         ab = array([[4.0, 4.0, 4.0, 4.0],
158:                     [1.0, 0.5, 0.2j, -1.0]])
159:         c = cholesky_banded(ab, lower=True)
160:         lfac = zeros_like(a)
161:         lfac[list(range(4)), list(range(4))] = c[0]
162:         lfac[(1, 2, 3), (0, 1, 2)] = c[1, :3]
163:         assert_array_almost_equal(a, dot(lfac, lfac.conj().T))
164: 
165:         b = array([0.0, 0.5j, 3.8j, 3.8])
166:         x = cho_solve_banded((c, True), b)
167:         assert_array_almost_equal(x, [0.0, 0.0, 1.0j, 1.0])
168: 
169: 
170: class TestOverwrite(object):
171:     def test_cholesky(self):
172:         assert_no_overwrite(cholesky, [(3, 3)])
173: 
174:     def test_cho_factor(self):
175:         assert_no_overwrite(cho_factor, [(3, 3)])
176: 
177:     def test_cho_solve(self):
178:         x = array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
179:         xcho = cho_factor(x)
180:         assert_no_overwrite(lambda b: cho_solve(xcho, b), [(3,)])
181: 
182:     def test_cholesky_banded(self):
183:         assert_no_overwrite(cholesky_banded, [(2, 3)])
184: 
185:     def test_cho_solve_banded(self):
186:         x = array([[0, -1, -1], [2, 2, 2]])
187:         xcho = cholesky_banded(x)
188:         assert_no_overwrite(lambda b: cho_solve_banded((xcho, False), b),
189:                             [(3,)])
190: 
191: 
192: class TestEmptyArray(object):
193:     def test_cho_factor_empty_square(self):
194:         a = empty((0, 0))
195:         b = array([])
196:         c = array([[]])
197:         d = []
198:         e = [[]]
199: 
200:         x, _ = cho_factor(a)
201:         assert_array_equal(x, a)
202: 
203:         for x in ([b, c, d, e]):
204:             assert_raises(ValueError, cho_factor, x)
205: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_array_equal' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_73301 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_73301) is not StypyTypeError):

    if (import_73301 != 'pyd_module'):
        __import__(import_73301)
        sys_modules_73302 = sys.modules[import_73301]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_73302.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_73302, sys_modules_73302.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_array_equal'], [assert_array_almost_equal, assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_73301)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from pytest import assert_raises' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_73303 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_73303) is not StypyTypeError):

    if (import_73303 != 'pyd_module'):
        __import__(import_73303)
        sys_modules_73304 = sys.modules[import_73303]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_73304.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_73304, sys_modules_73304.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_73303)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import array, transpose, dot, conjugate, zeros_like, empty' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_73305 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_73305) is not StypyTypeError):

    if (import_73305 != 'pyd_module'):
        __import__(import_73305)
        sys_modules_73306 = sys.modules[import_73305]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_73306.module_type_store, module_type_store, ['array', 'transpose', 'dot', 'conjugate', 'zeros_like', 'empty'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_73306, sys_modules_73306.module_type_store, module_type_store)
    else:
        from numpy import array, transpose, dot, conjugate, zeros_like, empty

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['array', 'transpose', 'dot', 'conjugate', 'zeros_like', 'empty'], [array, transpose, dot, conjugate, zeros_like, empty])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_73305)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.random import random' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_73307 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.random')

if (type(import_73307) is not StypyTypeError):

    if (import_73307 != 'pyd_module'):
        __import__(import_73307)
        sys_modules_73308 = sys.modules[import_73307]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.random', sys_modules_73308.module_type_store, module_type_store, ['random'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_73308, sys_modules_73308.module_type_store, module_type_store)
    else:
        from numpy.random import random

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.random', None, module_type_store, ['random'], [random])

else:
    # Assigning a type to the variable 'numpy.random' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.random', import_73307)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.linalg import cholesky, cholesky_banded, cho_solve_banded, cho_factor, cho_solve' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_73309 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg')

if (type(import_73309) is not StypyTypeError):

    if (import_73309 != 'pyd_module'):
        __import__(import_73309)
        sys_modules_73310 = sys.modules[import_73309]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', sys_modules_73310.module_type_store, module_type_store, ['cholesky', 'cholesky_banded', 'cho_solve_banded', 'cho_factor', 'cho_solve'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_73310, sys_modules_73310.module_type_store, module_type_store)
    else:
        from scipy.linalg import cholesky, cholesky_banded, cho_solve_banded, cho_factor, cho_solve

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', None, module_type_store, ['cholesky', 'cholesky_banded', 'cho_solve_banded', 'cho_factor', 'cho_solve'], [cholesky, cholesky_banded, cho_solve_banded, cho_factor, cho_solve])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', import_73309)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg._testutils import assert_no_overwrite' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_73311 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg._testutils')

if (type(import_73311) is not StypyTypeError):

    if (import_73311 != 'pyd_module'):
        __import__(import_73311)
        sys_modules_73312 = sys.modules[import_73311]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg._testutils', sys_modules_73312.module_type_store, module_type_store, ['assert_no_overwrite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_73312, sys_modules_73312.module_type_store, module_type_store)
    else:
        from scipy.linalg._testutils import assert_no_overwrite

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg._testutils', None, module_type_store, ['assert_no_overwrite'], [assert_no_overwrite])

else:
    # Assigning a type to the variable 'scipy.linalg._testutils' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg._testutils', import_73311)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

# Declaration of the 'TestCholesky' class

class TestCholesky(object, ):

    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholesky.test_simple.__dict__.__setitem__('stypy_localization', localization)
        TestCholesky.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholesky.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholesky.test_simple.__dict__.__setitem__('stypy_function_name', 'TestCholesky.test_simple')
        TestCholesky.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholesky.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholesky.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholesky.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholesky.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholesky.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholesky.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholesky.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a List to a Name (line 17):
        
        # Assigning a List to a Name (line 17):
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_73313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_73314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_73315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_73314, int_73315)
        # Adding element type (line 17)
        int_73316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_73314, int_73316)
        # Adding element type (line 17)
        int_73317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_73314, int_73317)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), list_73313, list_73314)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_73318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_73319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 24), list_73318, int_73319)
        # Adding element type (line 17)
        int_73320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 24), list_73318, int_73320)
        # Adding element type (line 17)
        int_73321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 24), list_73318, int_73321)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), list_73313, list_73318)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_73322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_73323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 35), list_73322, int_73323)
        # Adding element type (line 17)
        int_73324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 35), list_73322, int_73324)
        # Adding element type (line 17)
        int_73325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 35), list_73322, int_73325)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), list_73313, list_73322)
        
        # Assigning a type to the variable 'a' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'a', list_73313)
        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to cholesky(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'a' (line 18)
        a_73327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'a', False)
        # Processing the call keyword arguments (line 18)
        kwargs_73328 = {}
        # Getting the type of 'cholesky' (line 18)
        cholesky_73326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 18)
        cholesky_call_result_73329 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), cholesky_73326, *[a_73327], **kwargs_73328)
        
        # Assigning a type to the variable 'c' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'c', cholesky_call_result_73329)
        
        # Call to assert_array_almost_equal(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Call to dot(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Call to transpose(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'c' (line 19)
        c_73333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 48), 'c', False)
        # Processing the call keyword arguments (line 19)
        kwargs_73334 = {}
        # Getting the type of 'transpose' (line 19)
        transpose_73332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'transpose', False)
        # Calling transpose(args, kwargs) (line 19)
        transpose_call_result_73335 = invoke(stypy.reporting.localization.Localization(__file__, 19, 38), transpose_73332, *[c_73333], **kwargs_73334)
        
        # Getting the type of 'c' (line 19)
        c_73336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 52), 'c', False)
        # Processing the call keyword arguments (line 19)
        kwargs_73337 = {}
        # Getting the type of 'dot' (line 19)
        dot_73331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 34), 'dot', False)
        # Calling dot(args, kwargs) (line 19)
        dot_call_result_73338 = invoke(stypy.reporting.localization.Localization(__file__, 19, 34), dot_73331, *[transpose_call_result_73335, c_73336], **kwargs_73337)
        
        # Getting the type of 'a' (line 19)
        a_73339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 56), 'a', False)
        # Processing the call keyword arguments (line 19)
        kwargs_73340 = {}
        # Getting the type of 'assert_array_almost_equal' (line 19)
        assert_array_almost_equal_73330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 19)
        assert_array_almost_equal_call_result_73341 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), assert_array_almost_equal_73330, *[dot_call_result_73338, a_73339], **kwargs_73340)
        
        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to transpose(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'c' (line 20)
        c_73343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'c', False)
        # Processing the call keyword arguments (line 20)
        kwargs_73344 = {}
        # Getting the type of 'transpose' (line 20)
        transpose_73342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'transpose', False)
        # Calling transpose(args, kwargs) (line 20)
        transpose_call_result_73345 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), transpose_73342, *[c_73343], **kwargs_73344)
        
        # Assigning a type to the variable 'c' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'c', transpose_call_result_73345)
        
        # Assigning a Call to a Name (line 21):
        
        # Assigning a Call to a Name (line 21):
        
        # Call to dot(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'c' (line 21)
        c_73347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'c', False)
        
        # Call to transpose(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'c' (line 21)
        c_73349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 29), 'c', False)
        # Processing the call keyword arguments (line 21)
        kwargs_73350 = {}
        # Getting the type of 'transpose' (line 21)
        transpose_73348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'transpose', False)
        # Calling transpose(args, kwargs) (line 21)
        transpose_call_result_73351 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), transpose_73348, *[c_73349], **kwargs_73350)
        
        # Processing the call keyword arguments (line 21)
        kwargs_73352 = {}
        # Getting the type of 'dot' (line 21)
        dot_73346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'dot', False)
        # Calling dot(args, kwargs) (line 21)
        dot_call_result_73353 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), dot_73346, *[c_73347, transpose_call_result_73351], **kwargs_73352)
        
        # Assigning a type to the variable 'a' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'a', dot_call_result_73353)
        
        # Call to assert_array_almost_equal(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Call to cholesky(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'a' (line 22)
        a_73356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 43), 'a', False)
        # Processing the call keyword arguments (line 22)
        int_73357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 52), 'int')
        keyword_73358 = int_73357
        kwargs_73359 = {'lower': keyword_73358}
        # Getting the type of 'cholesky' (line 22)
        cholesky_73355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 22)
        cholesky_call_result_73360 = invoke(stypy.reporting.localization.Localization(__file__, 22, 34), cholesky_73355, *[a_73356], **kwargs_73359)
        
        # Getting the type of 'c' (line 22)
        c_73361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 56), 'c', False)
        # Processing the call keyword arguments (line 22)
        kwargs_73362 = {}
        # Getting the type of 'assert_array_almost_equal' (line 22)
        assert_array_almost_equal_73354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 22)
        assert_array_almost_equal_call_result_73363 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), assert_array_almost_equal_73354, *[cholesky_call_result_73360, c_73361], **kwargs_73362)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_73364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_73364


    @norecursion
    def test_check_finite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_finite'
        module_type_store = module_type_store.open_function_context('test_check_finite', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_localization', localization)
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_function_name', 'TestCholesky.test_check_finite')
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholesky.test_check_finite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholesky.test_check_finite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_finite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_finite(...)' code ##################

        
        # Assigning a List to a Name (line 25):
        
        # Assigning a List to a Name (line 25):
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_73365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_73366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_73367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_73366, int_73367)
        # Adding element type (line 25)
        int_73368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_73366, int_73368)
        # Adding element type (line 25)
        int_73369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_73366, int_73369)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_73365, list_73366)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_73370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_73371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_73370, int_73371)
        # Adding element type (line 25)
        int_73372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_73370, int_73372)
        # Adding element type (line 25)
        int_73373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_73370, int_73373)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_73365, list_73370)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_73374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_73375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 35), list_73374, int_73375)
        # Adding element type (line 25)
        int_73376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 35), list_73374, int_73376)
        # Adding element type (line 25)
        int_73377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 35), list_73374, int_73377)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_73365, list_73374)
        
        # Assigning a type to the variable 'a' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'a', list_73365)
        
        # Assigning a Call to a Name (line 26):
        
        # Assigning a Call to a Name (line 26):
        
        # Call to cholesky(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'a' (line 26)
        a_73379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'a', False)
        # Processing the call keyword arguments (line 26)
        # Getting the type of 'False' (line 26)
        False_73380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 37), 'False', False)
        keyword_73381 = False_73380
        kwargs_73382 = {'check_finite': keyword_73381}
        # Getting the type of 'cholesky' (line 26)
        cholesky_73378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 26)
        cholesky_call_result_73383 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), cholesky_73378, *[a_73379], **kwargs_73382)
        
        # Assigning a type to the variable 'c' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'c', cholesky_call_result_73383)
        
        # Call to assert_array_almost_equal(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to dot(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to transpose(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'c' (line 27)
        c_73387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 48), 'c', False)
        # Processing the call keyword arguments (line 27)
        kwargs_73388 = {}
        # Getting the type of 'transpose' (line 27)
        transpose_73386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 38), 'transpose', False)
        # Calling transpose(args, kwargs) (line 27)
        transpose_call_result_73389 = invoke(stypy.reporting.localization.Localization(__file__, 27, 38), transpose_73386, *[c_73387], **kwargs_73388)
        
        # Getting the type of 'c' (line 27)
        c_73390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 52), 'c', False)
        # Processing the call keyword arguments (line 27)
        kwargs_73391 = {}
        # Getting the type of 'dot' (line 27)
        dot_73385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'dot', False)
        # Calling dot(args, kwargs) (line 27)
        dot_call_result_73392 = invoke(stypy.reporting.localization.Localization(__file__, 27, 34), dot_73385, *[transpose_call_result_73389, c_73390], **kwargs_73391)
        
        # Getting the type of 'a' (line 27)
        a_73393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 56), 'a', False)
        # Processing the call keyword arguments (line 27)
        kwargs_73394 = {}
        # Getting the type of 'assert_array_almost_equal' (line 27)
        assert_array_almost_equal_73384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 27)
        assert_array_almost_equal_call_result_73395 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_array_almost_equal_73384, *[dot_call_result_73392, a_73393], **kwargs_73394)
        
        
        # Assigning a Call to a Name (line 28):
        
        # Assigning a Call to a Name (line 28):
        
        # Call to transpose(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'c' (line 28)
        c_73397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'c', False)
        # Processing the call keyword arguments (line 28)
        kwargs_73398 = {}
        # Getting the type of 'transpose' (line 28)
        transpose_73396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'transpose', False)
        # Calling transpose(args, kwargs) (line 28)
        transpose_call_result_73399 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), transpose_73396, *[c_73397], **kwargs_73398)
        
        # Assigning a type to the variable 'c' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'c', transpose_call_result_73399)
        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Call to dot(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'c' (line 29)
        c_73401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'c', False)
        
        # Call to transpose(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'c' (line 29)
        c_73403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'c', False)
        # Processing the call keyword arguments (line 29)
        kwargs_73404 = {}
        # Getting the type of 'transpose' (line 29)
        transpose_73402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'transpose', False)
        # Calling transpose(args, kwargs) (line 29)
        transpose_call_result_73405 = invoke(stypy.reporting.localization.Localization(__file__, 29, 19), transpose_73402, *[c_73403], **kwargs_73404)
        
        # Processing the call keyword arguments (line 29)
        kwargs_73406 = {}
        # Getting the type of 'dot' (line 29)
        dot_73400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'dot', False)
        # Calling dot(args, kwargs) (line 29)
        dot_call_result_73407 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), dot_73400, *[c_73401, transpose_call_result_73405], **kwargs_73406)
        
        # Assigning a type to the variable 'a' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'a', dot_call_result_73407)
        
        # Call to assert_array_almost_equal(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to cholesky(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'a' (line 30)
        a_73410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 'a', False)
        # Processing the call keyword arguments (line 30)
        int_73411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 52), 'int')
        keyword_73412 = int_73411
        # Getting the type of 'False' (line 30)
        False_73413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 68), 'False', False)
        keyword_73414 = False_73413
        kwargs_73415 = {'lower': keyword_73412, 'check_finite': keyword_73414}
        # Getting the type of 'cholesky' (line 30)
        cholesky_73409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 30)
        cholesky_call_result_73416 = invoke(stypy.reporting.localization.Localization(__file__, 30, 34), cholesky_73409, *[a_73410], **kwargs_73415)
        
        # Getting the type of 'c' (line 30)
        c_73417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 76), 'c', False)
        # Processing the call keyword arguments (line 30)
        kwargs_73418 = {}
        # Getting the type of 'assert_array_almost_equal' (line 30)
        assert_array_almost_equal_73408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 30)
        assert_array_almost_equal_call_result_73419 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_array_almost_equal_73408, *[cholesky_call_result_73416, c_73417], **kwargs_73418)
        
        
        # ################# End of 'test_check_finite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_finite' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_73420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73420)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_finite'
        return stypy_return_type_73420


    @norecursion
    def test_simple_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_complex'
        module_type_store = module_type_store.open_function_context('test_simple_complex', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_localization', localization)
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_function_name', 'TestCholesky.test_simple_complex')
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholesky.test_simple_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholesky.test_simple_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to array(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_73422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_73423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        int_73424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
        complex_73425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'complex')
        # Applying the binary operator '+' (line 33)
        result_add_73426 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 20), '+', int_73424, complex_73425)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), list_73423, result_add_73426)
        # Adding element type (line 33)
        int_73427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 26), 'int')
        complex_73428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'complex')
        # Applying the binary operator '+' (line 33)
        result_add_73429 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 26), '+', int_73427, complex_73428)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), list_73423, result_add_73429)
        # Adding element type (line 33)
        int_73430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), list_73423, int_73430)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_73422, list_73423)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_73431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        int_73432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 36), list_73431, int_73432)
        # Adding element type (line 33)
        int_73433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 40), 'int')
        complex_73434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 42), 'complex')
        # Applying the binary operator '+' (line 33)
        result_add_73435 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 40), '+', int_73433, complex_73434)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 36), list_73431, result_add_73435)
        # Adding element type (line 33)
        int_73436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 46), 'int')
        complex_73437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 48), 'complex')
        # Applying the binary operator '+' (line 33)
        result_add_73438 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 46), '+', int_73436, complex_73437)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 36), list_73431, result_add_73438)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_73422, list_73431)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_73439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        int_73440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 53), list_73439, int_73440)
        # Adding element type (line 33)
        int_73441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 53), list_73439, int_73441)
        # Adding element type (line 33)
        int_73442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 60), 'int')
        complex_73443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 62), 'complex')
        # Applying the binary operator '+' (line 33)
        result_add_73444 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 60), '+', int_73442, complex_73443)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 53), list_73439, result_add_73444)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 18), list_73422, list_73439)
        
        # Processing the call keyword arguments (line 33)
        kwargs_73445 = {}
        # Getting the type of 'array' (line 33)
        array_73421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'array', False)
        # Calling array(args, kwargs) (line 33)
        array_call_result_73446 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), array_73421, *[list_73422], **kwargs_73445)
        
        # Assigning a type to the variable 'm' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'm', array_call_result_73446)
        
        # Assigning a Call to a Name (line 34):
        
        # Assigning a Call to a Name (line 34):
        
        # Call to dot(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to transpose(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to conjugate(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'm' (line 34)
        m_73450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 36), 'm', False)
        # Processing the call keyword arguments (line 34)
        kwargs_73451 = {}
        # Getting the type of 'conjugate' (line 34)
        conjugate_73449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'conjugate', False)
        # Calling conjugate(args, kwargs) (line 34)
        conjugate_call_result_73452 = invoke(stypy.reporting.localization.Localization(__file__, 34, 26), conjugate_73449, *[m_73450], **kwargs_73451)
        
        # Processing the call keyword arguments (line 34)
        kwargs_73453 = {}
        # Getting the type of 'transpose' (line 34)
        transpose_73448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'transpose', False)
        # Calling transpose(args, kwargs) (line 34)
        transpose_call_result_73454 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), transpose_73448, *[conjugate_call_result_73452], **kwargs_73453)
        
        # Getting the type of 'm' (line 34)
        m_73455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 41), 'm', False)
        # Processing the call keyword arguments (line 34)
        kwargs_73456 = {}
        # Getting the type of 'dot' (line 34)
        dot_73447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'dot', False)
        # Calling dot(args, kwargs) (line 34)
        dot_call_result_73457 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), dot_73447, *[transpose_call_result_73454, m_73455], **kwargs_73456)
        
        # Assigning a type to the variable 'a' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'a', dot_call_result_73457)
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to cholesky(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'a' (line 35)
        a_73459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'a', False)
        # Processing the call keyword arguments (line 35)
        kwargs_73460 = {}
        # Getting the type of 'cholesky' (line 35)
        cholesky_73458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 35)
        cholesky_call_result_73461 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), cholesky_73458, *[a_73459], **kwargs_73460)
        
        # Assigning a type to the variable 'c' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'c', cholesky_call_result_73461)
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to dot(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Call to transpose(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Call to conjugate(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'c' (line 36)
        c_73465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'c', False)
        # Processing the call keyword arguments (line 36)
        kwargs_73466 = {}
        # Getting the type of 'conjugate' (line 36)
        conjugate_73464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'conjugate', False)
        # Calling conjugate(args, kwargs) (line 36)
        conjugate_call_result_73467 = invoke(stypy.reporting.localization.Localization(__file__, 36, 27), conjugate_73464, *[c_73465], **kwargs_73466)
        
        # Processing the call keyword arguments (line 36)
        kwargs_73468 = {}
        # Getting the type of 'transpose' (line 36)
        transpose_73463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'transpose', False)
        # Calling transpose(args, kwargs) (line 36)
        transpose_call_result_73469 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), transpose_73463, *[conjugate_call_result_73467], **kwargs_73468)
        
        # Getting the type of 'c' (line 36)
        c_73470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 42), 'c', False)
        # Processing the call keyword arguments (line 36)
        kwargs_73471 = {}
        # Getting the type of 'dot' (line 36)
        dot_73462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'dot', False)
        # Calling dot(args, kwargs) (line 36)
        dot_call_result_73472 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), dot_73462, *[transpose_call_result_73469, c_73470], **kwargs_73471)
        
        # Assigning a type to the variable 'a1' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'a1', dot_call_result_73472)
        
        # Call to assert_array_almost_equal(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'a' (line 37)
        a_73474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'a', False)
        # Getting the type of 'a1' (line 37)
        a1_73475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 37), 'a1', False)
        # Processing the call keyword arguments (line 37)
        kwargs_73476 = {}
        # Getting the type of 'assert_array_almost_equal' (line 37)
        assert_array_almost_equal_73473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 37)
        assert_array_almost_equal_call_result_73477 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assert_array_almost_equal_73473, *[a_73474, a1_73475], **kwargs_73476)
        
        
        # Assigning a Call to a Name (line 38):
        
        # Assigning a Call to a Name (line 38):
        
        # Call to transpose(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'c' (line 38)
        c_73479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'c', False)
        # Processing the call keyword arguments (line 38)
        kwargs_73480 = {}
        # Getting the type of 'transpose' (line 38)
        transpose_73478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'transpose', False)
        # Calling transpose(args, kwargs) (line 38)
        transpose_call_result_73481 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), transpose_73478, *[c_73479], **kwargs_73480)
        
        # Assigning a type to the variable 'c' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'c', transpose_call_result_73481)
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to dot(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'c' (line 39)
        c_73483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'c', False)
        
        # Call to transpose(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to conjugate(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'c' (line 39)
        c_73486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'c', False)
        # Processing the call keyword arguments (line 39)
        kwargs_73487 = {}
        # Getting the type of 'conjugate' (line 39)
        conjugate_73485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'conjugate', False)
        # Calling conjugate(args, kwargs) (line 39)
        conjugate_call_result_73488 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), conjugate_73485, *[c_73486], **kwargs_73487)
        
        # Processing the call keyword arguments (line 39)
        kwargs_73489 = {}
        # Getting the type of 'transpose' (line 39)
        transpose_73484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'transpose', False)
        # Calling transpose(args, kwargs) (line 39)
        transpose_call_result_73490 = invoke(stypy.reporting.localization.Localization(__file__, 39, 19), transpose_73484, *[conjugate_call_result_73488], **kwargs_73489)
        
        # Processing the call keyword arguments (line 39)
        kwargs_73491 = {}
        # Getting the type of 'dot' (line 39)
        dot_73482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'dot', False)
        # Calling dot(args, kwargs) (line 39)
        dot_call_result_73492 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), dot_73482, *[c_73483, transpose_call_result_73490], **kwargs_73491)
        
        # Assigning a type to the variable 'a' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'a', dot_call_result_73492)
        
        # Call to assert_array_almost_equal(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to cholesky(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'a' (line 40)
        a_73495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'a', False)
        # Processing the call keyword arguments (line 40)
        int_73496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 52), 'int')
        keyword_73497 = int_73496
        kwargs_73498 = {'lower': keyword_73497}
        # Getting the type of 'cholesky' (line 40)
        cholesky_73494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 40)
        cholesky_call_result_73499 = invoke(stypy.reporting.localization.Localization(__file__, 40, 34), cholesky_73494, *[a_73495], **kwargs_73498)
        
        # Getting the type of 'c' (line 40)
        c_73500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 56), 'c', False)
        # Processing the call keyword arguments (line 40)
        kwargs_73501 = {}
        # Getting the type of 'assert_array_almost_equal' (line 40)
        assert_array_almost_equal_73493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 40)
        assert_array_almost_equal_call_result_73502 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assert_array_almost_equal_73493, *[cholesky_call_result_73499, c_73500], **kwargs_73501)
        
        
        # ################# End of 'test_simple_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_73503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_complex'
        return stypy_return_type_73503


    @norecursion
    def test_random(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random'
        module_type_store = module_type_store.open_function_context('test_random', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholesky.test_random.__dict__.__setitem__('stypy_localization', localization)
        TestCholesky.test_random.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholesky.test_random.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholesky.test_random.__dict__.__setitem__('stypy_function_name', 'TestCholesky.test_random')
        TestCholesky.test_random.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholesky.test_random.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholesky.test_random.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholesky.test_random.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholesky.test_random.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholesky.test_random.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholesky.test_random.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholesky.test_random', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random(...)' code ##################

        
        # Assigning a Num to a Name (line 43):
        
        # Assigning a Num to a Name (line 43):
        int_73504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 12), 'int')
        # Assigning a type to the variable 'n' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'n', int_73504)
        
        
        # Call to range(...): (line 44)
        # Processing the call arguments (line 44)
        int_73506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_73507 = {}
        # Getting the type of 'range' (line 44)
        range_73505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'range', False)
        # Calling range(args, kwargs) (line 44)
        range_call_result_73508 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), range_73505, *[int_73506], **kwargs_73507)
        
        # Testing the type of a for loop iterable (line 44)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 8), range_call_result_73508)
        # Getting the type of the for loop variable (line 44)
        for_loop_var_73509 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 8), range_call_result_73508)
        # Assigning a type to the variable 'k' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'k', for_loop_var_73509)
        # SSA begins for a for statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to random(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_73511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        # Adding element type (line 45)
        # Getting the type of 'n' (line 45)
        n_73512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 23), list_73511, n_73512)
        # Adding element type (line 45)
        # Getting the type of 'n' (line 45)
        n_73513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 23), list_73511, n_73513)
        
        # Processing the call keyword arguments (line 45)
        kwargs_73514 = {}
        # Getting the type of 'random' (line 45)
        random_73510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'random', False)
        # Calling random(args, kwargs) (line 45)
        random_call_result_73515 = invoke(stypy.reporting.localization.Localization(__file__, 45, 16), random_73510, *[list_73511], **kwargs_73514)
        
        # Assigning a type to the variable 'm' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'm', random_call_result_73515)
        
        
        # Call to range(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'n' (line 46)
        n_73517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'n', False)
        # Processing the call keyword arguments (line 46)
        kwargs_73518 = {}
        # Getting the type of 'range' (line 46)
        range_73516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'range', False)
        # Calling range(args, kwargs) (line 46)
        range_call_result_73519 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), range_73516, *[n_73517], **kwargs_73518)
        
        # Testing the type of a for loop iterable (line 46)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 46, 12), range_call_result_73519)
        # Getting the type of the for loop variable (line 46)
        for_loop_var_73520 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 46, 12), range_call_result_73519)
        # Assigning a type to the variable 'i' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'i', for_loop_var_73520)
        # SSA begins for a for statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 47):
        
        # Assigning a BinOp to a Subscript (line 47):
        int_73521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'int')
        float_73522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'float')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_73523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        # Getting the type of 'i' (line 47)
        i_73524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 35), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 35), tuple_73523, i_73524)
        # Adding element type (line 47)
        # Getting the type of 'i' (line 47)
        i_73525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 38), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 35), tuple_73523, i_73525)
        
        # Getting the type of 'm' (line 47)
        m_73526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'm')
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___73527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 33), m_73526, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_73528 = invoke(stypy.reporting.localization.Localization(__file__, 47, 33), getitem___73527, tuple_73523)
        
        # Applying the binary operator '+' (line 47)
        result_add_73529 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 30), '+', float_73522, subscript_call_result_73528)
        
        # Applying the binary operator '*' (line 47)
        result_mul_73530 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 26), '*', int_73521, result_add_73529)
        
        # Getting the type of 'm' (line 47)
        m_73531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'm')
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_73532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        # Getting the type of 'i' (line 47)
        i_73533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 18), tuple_73532, i_73533)
        # Adding element type (line 47)
        # Getting the type of 'i' (line 47)
        i_73534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 18), tuple_73532, i_73534)
        
        # Storing an element on a container (line 47)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 16), m_73531, (tuple_73532, result_mul_73530))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to dot(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to transpose(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'm' (line 48)
        m_73537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'm', False)
        # Processing the call keyword arguments (line 48)
        kwargs_73538 = {}
        # Getting the type of 'transpose' (line 48)
        transpose_73536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'transpose', False)
        # Calling transpose(args, kwargs) (line 48)
        transpose_call_result_73539 = invoke(stypy.reporting.localization.Localization(__file__, 48, 20), transpose_73536, *[m_73537], **kwargs_73538)
        
        # Getting the type of 'm' (line 48)
        m_73540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'm', False)
        # Processing the call keyword arguments (line 48)
        kwargs_73541 = {}
        # Getting the type of 'dot' (line 48)
        dot_73535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'dot', False)
        # Calling dot(args, kwargs) (line 48)
        dot_call_result_73542 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), dot_73535, *[transpose_call_result_73539, m_73540], **kwargs_73541)
        
        # Assigning a type to the variable 'a' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'a', dot_call_result_73542)
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to cholesky(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'a' (line 49)
        a_73544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'a', False)
        # Processing the call keyword arguments (line 49)
        kwargs_73545 = {}
        # Getting the type of 'cholesky' (line 49)
        cholesky_73543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 49)
        cholesky_call_result_73546 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), cholesky_73543, *[a_73544], **kwargs_73545)
        
        # Assigning a type to the variable 'c' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'c', cholesky_call_result_73546)
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to dot(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to transpose(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'c' (line 50)
        c_73549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'c', False)
        # Processing the call keyword arguments (line 50)
        kwargs_73550 = {}
        # Getting the type of 'transpose' (line 50)
        transpose_73548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'transpose', False)
        # Calling transpose(args, kwargs) (line 50)
        transpose_call_result_73551 = invoke(stypy.reporting.localization.Localization(__file__, 50, 21), transpose_73548, *[c_73549], **kwargs_73550)
        
        # Getting the type of 'c' (line 50)
        c_73552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'c', False)
        # Processing the call keyword arguments (line 50)
        kwargs_73553 = {}
        # Getting the type of 'dot' (line 50)
        dot_73547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'dot', False)
        # Calling dot(args, kwargs) (line 50)
        dot_call_result_73554 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), dot_73547, *[transpose_call_result_73551, c_73552], **kwargs_73553)
        
        # Assigning a type to the variable 'a1' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'a1', dot_call_result_73554)
        
        # Call to assert_array_almost_equal(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'a' (line 51)
        a_73556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 38), 'a', False)
        # Getting the type of 'a1' (line 51)
        a1_73557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 41), 'a1', False)
        # Processing the call keyword arguments (line 51)
        kwargs_73558 = {}
        # Getting the type of 'assert_array_almost_equal' (line 51)
        assert_array_almost_equal_73555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 51)
        assert_array_almost_equal_call_result_73559 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), assert_array_almost_equal_73555, *[a_73556, a1_73557], **kwargs_73558)
        
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to transpose(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'c' (line 52)
        c_73561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'c', False)
        # Processing the call keyword arguments (line 52)
        kwargs_73562 = {}
        # Getting the type of 'transpose' (line 52)
        transpose_73560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'transpose', False)
        # Calling transpose(args, kwargs) (line 52)
        transpose_call_result_73563 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), transpose_73560, *[c_73561], **kwargs_73562)
        
        # Assigning a type to the variable 'c' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'c', transpose_call_result_73563)
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to dot(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'c' (line 53)
        c_73565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'c', False)
        
        # Call to transpose(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'c' (line 53)
        c_73567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'c', False)
        # Processing the call keyword arguments (line 53)
        kwargs_73568 = {}
        # Getting the type of 'transpose' (line 53)
        transpose_73566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'transpose', False)
        # Calling transpose(args, kwargs) (line 53)
        transpose_call_result_73569 = invoke(stypy.reporting.localization.Localization(__file__, 53, 23), transpose_73566, *[c_73567], **kwargs_73568)
        
        # Processing the call keyword arguments (line 53)
        kwargs_73570 = {}
        # Getting the type of 'dot' (line 53)
        dot_73564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'dot', False)
        # Calling dot(args, kwargs) (line 53)
        dot_call_result_73571 = invoke(stypy.reporting.localization.Localization(__file__, 53, 16), dot_73564, *[c_73565, transpose_call_result_73569], **kwargs_73570)
        
        # Assigning a type to the variable 'a' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'a', dot_call_result_73571)
        
        # Call to assert_array_almost_equal(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to cholesky(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'a' (line 54)
        a_73574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 47), 'a', False)
        # Processing the call keyword arguments (line 54)
        int_73575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 56), 'int')
        keyword_73576 = int_73575
        kwargs_73577 = {'lower': keyword_73576}
        # Getting the type of 'cholesky' (line 54)
        cholesky_73573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 54)
        cholesky_call_result_73578 = invoke(stypy.reporting.localization.Localization(__file__, 54, 38), cholesky_73573, *[a_73574], **kwargs_73577)
        
        # Getting the type of 'c' (line 54)
        c_73579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 60), 'c', False)
        # Processing the call keyword arguments (line 54)
        kwargs_73580 = {}
        # Getting the type of 'assert_array_almost_equal' (line 54)
        assert_array_almost_equal_73572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 54)
        assert_array_almost_equal_call_result_73581 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), assert_array_almost_equal_73572, *[cholesky_call_result_73578, c_73579], **kwargs_73580)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_73582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random'
        return stypy_return_type_73582


    @norecursion
    def test_random_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_complex'
        module_type_store = module_type_store.open_function_context('test_random_complex', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_localization', localization)
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_function_name', 'TestCholesky.test_random_complex')
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholesky.test_random_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholesky.test_random_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_complex(...)' code ##################

        
        # Assigning a Num to a Name (line 57):
        
        # Assigning a Num to a Name (line 57):
        int_73583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'int')
        # Assigning a type to the variable 'n' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'n', int_73583)
        
        
        # Call to range(...): (line 58)
        # Processing the call arguments (line 58)
        int_73585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'int')
        # Processing the call keyword arguments (line 58)
        kwargs_73586 = {}
        # Getting the type of 'range' (line 58)
        range_73584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'range', False)
        # Calling range(args, kwargs) (line 58)
        range_call_result_73587 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), range_73584, *[int_73585], **kwargs_73586)
        
        # Testing the type of a for loop iterable (line 58)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 8), range_call_result_73587)
        # Getting the type of the for loop variable (line 58)
        for_loop_var_73588 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 8), range_call_result_73587)
        # Assigning a type to the variable 'k' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'k', for_loop_var_73588)
        # SSA begins for a for statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 59):
        
        # Assigning a BinOp to a Name (line 59):
        
        # Call to random(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_73590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        # Getting the type of 'n' (line 59)
        n_73591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 23), list_73590, n_73591)
        # Adding element type (line 59)
        # Getting the type of 'n' (line 59)
        n_73592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 23), list_73590, n_73592)
        
        # Processing the call keyword arguments (line 59)
        kwargs_73593 = {}
        # Getting the type of 'random' (line 59)
        random_73589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'random', False)
        # Calling random(args, kwargs) (line 59)
        random_call_result_73594 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), random_73589, *[list_73590], **kwargs_73593)
        
        complex_73595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'complex')
        
        # Call to random(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_73597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        # Getting the type of 'n' (line 59)
        n_73598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 42), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), list_73597, n_73598)
        # Adding element type (line 59)
        # Getting the type of 'n' (line 59)
        n_73599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), list_73597, n_73599)
        
        # Processing the call keyword arguments (line 59)
        kwargs_73600 = {}
        # Getting the type of 'random' (line 59)
        random_73596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'random', False)
        # Calling random(args, kwargs) (line 59)
        random_call_result_73601 = invoke(stypy.reporting.localization.Localization(__file__, 59, 34), random_73596, *[list_73597], **kwargs_73600)
        
        # Applying the binary operator '*' (line 59)
        result_mul_73602 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 31), '*', complex_73595, random_call_result_73601)
        
        # Applying the binary operator '+' (line 59)
        result_add_73603 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 16), '+', random_call_result_73594, result_mul_73602)
        
        # Assigning a type to the variable 'm' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'm', result_add_73603)
        
        
        # Call to range(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'n' (line 60)
        n_73605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'n', False)
        # Processing the call keyword arguments (line 60)
        kwargs_73606 = {}
        # Getting the type of 'range' (line 60)
        range_73604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'range', False)
        # Calling range(args, kwargs) (line 60)
        range_call_result_73607 = invoke(stypy.reporting.localization.Localization(__file__, 60, 21), range_73604, *[n_73605], **kwargs_73606)
        
        # Testing the type of a for loop iterable (line 60)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 60, 12), range_call_result_73607)
        # Getting the type of the for loop variable (line 60)
        for_loop_var_73608 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 60, 12), range_call_result_73607)
        # Assigning a type to the variable 'i' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'i', for_loop_var_73608)
        # SSA begins for a for statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 61):
        
        # Assigning a BinOp to a Subscript (line 61):
        int_73609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'int')
        float_73610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 30), 'float')
        
        # Call to abs(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 61)
        tuple_73612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 61)
        # Adding element type (line 61)
        # Getting the type of 'i' (line 61)
        i_73613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 39), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 39), tuple_73612, i_73613)
        # Adding element type (line 61)
        # Getting the type of 'i' (line 61)
        i_73614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 39), tuple_73612, i_73614)
        
        # Getting the type of 'm' (line 61)
        m_73615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 37), 'm', False)
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___73616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 37), m_73615, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_73617 = invoke(stypy.reporting.localization.Localization(__file__, 61, 37), getitem___73616, tuple_73612)
        
        # Processing the call keyword arguments (line 61)
        kwargs_73618 = {}
        # Getting the type of 'abs' (line 61)
        abs_73611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'abs', False)
        # Calling abs(args, kwargs) (line 61)
        abs_call_result_73619 = invoke(stypy.reporting.localization.Localization(__file__, 61, 33), abs_73611, *[subscript_call_result_73617], **kwargs_73618)
        
        # Applying the binary operator '+' (line 61)
        result_add_73620 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 30), '+', float_73610, abs_call_result_73619)
        
        # Applying the binary operator '*' (line 61)
        result_mul_73621 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 26), '*', int_73609, result_add_73620)
        
        # Getting the type of 'm' (line 61)
        m_73622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'm')
        
        # Obtaining an instance of the builtin type 'tuple' (line 61)
        tuple_73623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 61)
        # Adding element type (line 61)
        # Getting the type of 'i' (line 61)
        i_73624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), tuple_73623, i_73624)
        # Adding element type (line 61)
        # Getting the type of 'i' (line 61)
        i_73625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), tuple_73623, i_73625)
        
        # Storing an element on a container (line 61)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), m_73622, (tuple_73623, result_mul_73621))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 62):
        
        # Assigning a Call to a Name (line 62):
        
        # Call to dot(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to transpose(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to conjugate(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'm' (line 62)
        m_73629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 40), 'm', False)
        # Processing the call keyword arguments (line 62)
        kwargs_73630 = {}
        # Getting the type of 'conjugate' (line 62)
        conjugate_73628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'conjugate', False)
        # Calling conjugate(args, kwargs) (line 62)
        conjugate_call_result_73631 = invoke(stypy.reporting.localization.Localization(__file__, 62, 30), conjugate_73628, *[m_73629], **kwargs_73630)
        
        # Processing the call keyword arguments (line 62)
        kwargs_73632 = {}
        # Getting the type of 'transpose' (line 62)
        transpose_73627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'transpose', False)
        # Calling transpose(args, kwargs) (line 62)
        transpose_call_result_73633 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), transpose_73627, *[conjugate_call_result_73631], **kwargs_73632)
        
        # Getting the type of 'm' (line 62)
        m_73634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 45), 'm', False)
        # Processing the call keyword arguments (line 62)
        kwargs_73635 = {}
        # Getting the type of 'dot' (line 62)
        dot_73626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'dot', False)
        # Calling dot(args, kwargs) (line 62)
        dot_call_result_73636 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), dot_73626, *[transpose_call_result_73633, m_73634], **kwargs_73635)
        
        # Assigning a type to the variable 'a' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'a', dot_call_result_73636)
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to cholesky(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'a' (line 63)
        a_73638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'a', False)
        # Processing the call keyword arguments (line 63)
        kwargs_73639 = {}
        # Getting the type of 'cholesky' (line 63)
        cholesky_73637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 63)
        cholesky_call_result_73640 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), cholesky_73637, *[a_73638], **kwargs_73639)
        
        # Assigning a type to the variable 'c' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'c', cholesky_call_result_73640)
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to dot(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to transpose(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to conjugate(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'c' (line 64)
        c_73644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'c', False)
        # Processing the call keyword arguments (line 64)
        kwargs_73645 = {}
        # Getting the type of 'conjugate' (line 64)
        conjugate_73643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'conjugate', False)
        # Calling conjugate(args, kwargs) (line 64)
        conjugate_call_result_73646 = invoke(stypy.reporting.localization.Localization(__file__, 64, 31), conjugate_73643, *[c_73644], **kwargs_73645)
        
        # Processing the call keyword arguments (line 64)
        kwargs_73647 = {}
        # Getting the type of 'transpose' (line 64)
        transpose_73642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'transpose', False)
        # Calling transpose(args, kwargs) (line 64)
        transpose_call_result_73648 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), transpose_73642, *[conjugate_call_result_73646], **kwargs_73647)
        
        # Getting the type of 'c' (line 64)
        c_73649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 46), 'c', False)
        # Processing the call keyword arguments (line 64)
        kwargs_73650 = {}
        # Getting the type of 'dot' (line 64)
        dot_73641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'dot', False)
        # Calling dot(args, kwargs) (line 64)
        dot_call_result_73651 = invoke(stypy.reporting.localization.Localization(__file__, 64, 17), dot_73641, *[transpose_call_result_73648, c_73649], **kwargs_73650)
        
        # Assigning a type to the variable 'a1' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'a1', dot_call_result_73651)
        
        # Call to assert_array_almost_equal(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'a' (line 65)
        a_73653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 38), 'a', False)
        # Getting the type of 'a1' (line 65)
        a1_73654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), 'a1', False)
        # Processing the call keyword arguments (line 65)
        kwargs_73655 = {}
        # Getting the type of 'assert_array_almost_equal' (line 65)
        assert_array_almost_equal_73652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 65)
        assert_array_almost_equal_call_result_73656 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), assert_array_almost_equal_73652, *[a_73653, a1_73654], **kwargs_73655)
        
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to transpose(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'c' (line 66)
        c_73658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 26), 'c', False)
        # Processing the call keyword arguments (line 66)
        kwargs_73659 = {}
        # Getting the type of 'transpose' (line 66)
        transpose_73657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'transpose', False)
        # Calling transpose(args, kwargs) (line 66)
        transpose_call_result_73660 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), transpose_73657, *[c_73658], **kwargs_73659)
        
        # Assigning a type to the variable 'c' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'c', transpose_call_result_73660)
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to dot(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'c' (line 67)
        c_73662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'c', False)
        
        # Call to transpose(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to conjugate(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'c' (line 67)
        c_73665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'c', False)
        # Processing the call keyword arguments (line 67)
        kwargs_73666 = {}
        # Getting the type of 'conjugate' (line 67)
        conjugate_73664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'conjugate', False)
        # Calling conjugate(args, kwargs) (line 67)
        conjugate_call_result_73667 = invoke(stypy.reporting.localization.Localization(__file__, 67, 33), conjugate_73664, *[c_73665], **kwargs_73666)
        
        # Processing the call keyword arguments (line 67)
        kwargs_73668 = {}
        # Getting the type of 'transpose' (line 67)
        transpose_73663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'transpose', False)
        # Calling transpose(args, kwargs) (line 67)
        transpose_call_result_73669 = invoke(stypy.reporting.localization.Localization(__file__, 67, 23), transpose_73663, *[conjugate_call_result_73667], **kwargs_73668)
        
        # Processing the call keyword arguments (line 67)
        kwargs_73670 = {}
        # Getting the type of 'dot' (line 67)
        dot_73661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'dot', False)
        # Calling dot(args, kwargs) (line 67)
        dot_call_result_73671 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), dot_73661, *[c_73662, transpose_call_result_73669], **kwargs_73670)
        
        # Assigning a type to the variable 'a' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'a', dot_call_result_73671)
        
        # Call to assert_array_almost_equal(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to cholesky(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'a' (line 68)
        a_73674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 47), 'a', False)
        # Processing the call keyword arguments (line 68)
        int_73675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 56), 'int')
        keyword_73676 = int_73675
        kwargs_73677 = {'lower': keyword_73676}
        # Getting the type of 'cholesky' (line 68)
        cholesky_73673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 68)
        cholesky_call_result_73678 = invoke(stypy.reporting.localization.Localization(__file__, 68, 38), cholesky_73673, *[a_73674], **kwargs_73677)
        
        # Getting the type of 'c' (line 68)
        c_73679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 60), 'c', False)
        # Processing the call keyword arguments (line 68)
        kwargs_73680 = {}
        # Getting the type of 'assert_array_almost_equal' (line 68)
        assert_array_almost_equal_73672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 68)
        assert_array_almost_equal_call_result_73681 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), assert_array_almost_equal_73672, *[cholesky_call_result_73678, c_73679], **kwargs_73680)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_73682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_complex'
        return stypy_return_type_73682


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholesky.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCholesky' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TestCholesky', TestCholesky)
# Declaration of the 'TestCholeskyBanded' class

class TestCholeskyBanded(object, ):
    str_73683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'Tests for cholesky_banded() and cho_solve_banded.')

    @norecursion
    def test_check_finite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_finite'
        module_type_store = module_type_store.open_function_context('test_check_finite', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_localization', localization)
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_function_name', 'TestCholeskyBanded.test_check_finite')
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholeskyBanded.test_check_finite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholeskyBanded.test_check_finite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_finite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_finite(...)' code ##################

        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to array(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_73685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_73686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        float_73687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 19), list_73686, float_73687)
        # Adding element type (line 76)
        float_73688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 19), list_73686, float_73688)
        # Adding element type (line 76)
        float_73689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 19), list_73686, float_73689)
        # Adding element type (line 76)
        float_73690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 19), list_73686, float_73690)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 18), list_73685, list_73686)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_73691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        float_73692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 19), list_73691, float_73692)
        # Adding element type (line 77)
        float_73693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 19), list_73691, float_73693)
        # Adding element type (line 77)
        float_73694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 19), list_73691, float_73694)
        # Adding element type (line 77)
        float_73695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 19), list_73691, float_73695)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 18), list_73685, list_73691)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_73696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        float_73697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_73696, float_73697)
        # Adding element type (line 78)
        float_73698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_73696, float_73698)
        # Adding element type (line 78)
        float_73699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_73696, float_73699)
        # Adding element type (line 78)
        float_73700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_73696, float_73700)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 18), list_73685, list_73696)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_73701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        float_73702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_73701, float_73702)
        # Adding element type (line 79)
        float_73703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_73701, float_73703)
        # Adding element type (line 79)
        float_73704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_73701, float_73704)
        # Adding element type (line 79)
        float_73705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_73701, float_73705)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 18), list_73685, list_73701)
        
        # Processing the call keyword arguments (line 76)
        kwargs_73706 = {}
        # Getting the type of 'array' (line 76)
        array_73684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'array', False)
        # Calling array(args, kwargs) (line 76)
        array_call_result_73707 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), array_73684, *[list_73685], **kwargs_73706)
        
        # Assigning a type to the variable 'a' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'a', array_call_result_73707)
        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to array(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_73709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_73710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        float_73711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), list_73710, float_73711)
        # Adding element type (line 81)
        float_73712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), list_73710, float_73712)
        # Adding element type (line 81)
        float_73713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), list_73710, float_73713)
        # Adding element type (line 81)
        float_73714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), list_73710, float_73714)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 19), list_73709, list_73710)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_73715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        float_73716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), list_73715, float_73716)
        # Adding element type (line 82)
        float_73717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), list_73715, float_73717)
        # Adding element type (line 82)
        float_73718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), list_73715, float_73718)
        # Adding element type (line 82)
        float_73719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), list_73715, float_73719)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 19), list_73709, list_73715)
        
        # Processing the call keyword arguments (line 81)
        kwargs_73720 = {}
        # Getting the type of 'array' (line 81)
        array_73708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'array', False)
        # Calling array(args, kwargs) (line 81)
        array_call_result_73721 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), array_73708, *[list_73709], **kwargs_73720)
        
        # Assigning a type to the variable 'ab' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'ab', array_call_result_73721)
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to cholesky_banded(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'ab' (line 83)
        ab_73723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'ab', False)
        # Processing the call keyword arguments (line 83)
        # Getting the type of 'False' (line 83)
        False_73724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 38), 'False', False)
        keyword_73725 = False_73724
        # Getting the type of 'False' (line 83)
        False_73726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 58), 'False', False)
        keyword_73727 = False_73726
        kwargs_73728 = {'lower': keyword_73725, 'check_finite': keyword_73727}
        # Getting the type of 'cholesky_banded' (line 83)
        cholesky_banded_73722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'cholesky_banded', False)
        # Calling cholesky_banded(args, kwargs) (line 83)
        cholesky_banded_call_result_73729 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), cholesky_banded_73722, *[ab_73723], **kwargs_73728)
        
        # Assigning a type to the variable 'c' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'c', cholesky_banded_call_result_73729)
        
        # Assigning a Call to a Name (line 84):
        
        # Assigning a Call to a Name (line 84):
        
        # Call to zeros_like(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'a' (line 84)
        a_73731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'a', False)
        # Processing the call keyword arguments (line 84)
        kwargs_73732 = {}
        # Getting the type of 'zeros_like' (line 84)
        zeros_like_73730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'zeros_like', False)
        # Calling zeros_like(args, kwargs) (line 84)
        zeros_like_call_result_73733 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), zeros_like_73730, *[a_73731], **kwargs_73732)
        
        # Assigning a type to the variable 'ufac' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'ufac', zeros_like_call_result_73733)
        
        # Assigning a Subscript to a Subscript (line 85):
        
        # Assigning a Subscript to a Subscript (line 85):
        
        # Obtaining the type of the subscript
        int_73734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 49), 'int')
        # Getting the type of 'c' (line 85)
        c_73735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 47), 'c')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___73736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 47), c_73735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_73737 = invoke(stypy.reporting.localization.Localization(__file__, 85, 47), getitem___73736, int_73734)
        
        # Getting the type of 'ufac' (line 85)
        ufac_73738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'ufac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_73739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        
        # Call to list(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to range(...): (line 85)
        # Processing the call arguments (line 85)
        int_73742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 24), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_73743 = {}
        # Getting the type of 'range' (line 85)
        range_73741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'range', False)
        # Calling range(args, kwargs) (line 85)
        range_call_result_73744 = invoke(stypy.reporting.localization.Localization(__file__, 85, 18), range_73741, *[int_73742], **kwargs_73743)
        
        # Processing the call keyword arguments (line 85)
        kwargs_73745 = {}
        # Getting the type of 'list' (line 85)
        list_73740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'list', False)
        # Calling list(args, kwargs) (line 85)
        list_call_result_73746 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), list_73740, *[range_call_result_73744], **kwargs_73745)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), tuple_73739, list_call_result_73746)
        # Adding element type (line 85)
        
        # Call to list(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to range(...): (line 85)
        # Processing the call arguments (line 85)
        int_73749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 40), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_73750 = {}
        # Getting the type of 'range' (line 85)
        range_73748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'range', False)
        # Calling range(args, kwargs) (line 85)
        range_call_result_73751 = invoke(stypy.reporting.localization.Localization(__file__, 85, 34), range_73748, *[int_73749], **kwargs_73750)
        
        # Processing the call keyword arguments (line 85)
        kwargs_73752 = {}
        # Getting the type of 'list' (line 85)
        list_73747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'list', False)
        # Calling list(args, kwargs) (line 85)
        list_call_result_73753 = invoke(stypy.reporting.localization.Localization(__file__, 85, 29), list_73747, *[range_call_result_73751], **kwargs_73752)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), tuple_73739, list_call_result_73753)
        
        # Storing an element on a container (line 85)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), ufac_73738, (tuple_73739, subscript_call_result_73737))
        
        # Assigning a Subscript to a Subscript (line 86):
        
        # Assigning a Subscript to a Subscript (line 86):
        
        # Obtaining the type of the subscript
        int_73754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 39), 'int')
        int_73755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 42), 'int')
        slice_73756 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 86, 37), int_73755, None, None)
        # Getting the type of 'c' (line 86)
        c_73757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'c')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___73758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 37), c_73757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_73759 = invoke(stypy.reporting.localization.Localization(__file__, 86, 37), getitem___73758, (int_73754, slice_73756))
        
        # Getting the type of 'ufac' (line 86)
        ufac_73760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'ufac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_73761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_73762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        int_73763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), tuple_73762, int_73763)
        # Adding element type (line 86)
        int_73764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), tuple_73762, int_73764)
        # Adding element type (line 86)
        int_73765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), tuple_73762, int_73765)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 13), tuple_73761, tuple_73762)
        # Adding element type (line 86)
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_73766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        int_73767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), tuple_73766, int_73767)
        # Adding element type (line 86)
        int_73768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), tuple_73766, int_73768)
        # Adding element type (line 86)
        int_73769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), tuple_73766, int_73769)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 13), tuple_73761, tuple_73766)
        
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), ufac_73760, (tuple_73761, subscript_call_result_73759))
        
        # Call to assert_array_almost_equal(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'a' (line 87)
        a_73771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'a', False)
        
        # Call to dot(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'ufac' (line 87)
        ufac_73773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'ufac', False)
        # Obtaining the member 'T' of a type (line 87)
        T_73774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 41), ufac_73773, 'T')
        # Getting the type of 'ufac' (line 87)
        ufac_73775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 49), 'ufac', False)
        # Processing the call keyword arguments (line 87)
        kwargs_73776 = {}
        # Getting the type of 'dot' (line 87)
        dot_73772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'dot', False)
        # Calling dot(args, kwargs) (line 87)
        dot_call_result_73777 = invoke(stypy.reporting.localization.Localization(__file__, 87, 37), dot_73772, *[T_73774, ufac_73775], **kwargs_73776)
        
        # Processing the call keyword arguments (line 87)
        kwargs_73778 = {}
        # Getting the type of 'assert_array_almost_equal' (line 87)
        assert_array_almost_equal_73770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 87)
        assert_array_almost_equal_call_result_73779 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), assert_array_almost_equal_73770, *[a_73771, dot_call_result_73777], **kwargs_73778)
        
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to array(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_73781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        float_73782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), list_73781, float_73782)
        # Adding element type (line 89)
        float_73783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), list_73781, float_73783)
        # Adding element type (line 89)
        float_73784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), list_73781, float_73784)
        # Adding element type (line 89)
        float_73785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 18), list_73781, float_73785)
        
        # Processing the call keyword arguments (line 89)
        kwargs_73786 = {}
        # Getting the type of 'array' (line 89)
        array_73780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'array', False)
        # Calling array(args, kwargs) (line 89)
        array_call_result_73787 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), array_73780, *[list_73781], **kwargs_73786)
        
        # Assigning a type to the variable 'b' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'b', array_call_result_73787)
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to cho_solve_banded(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining an instance of the builtin type 'tuple' (line 90)
        tuple_73789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 90)
        # Adding element type (line 90)
        # Getting the type of 'c' (line 90)
        c_73790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 30), tuple_73789, c_73790)
        # Adding element type (line 90)
        # Getting the type of 'False' (line 90)
        False_73791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 33), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 30), tuple_73789, False_73791)
        
        # Getting the type of 'b' (line 90)
        b_73792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 41), 'b', False)
        # Processing the call keyword arguments (line 90)
        # Getting the type of 'False' (line 90)
        False_73793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 57), 'False', False)
        keyword_73794 = False_73793
        kwargs_73795 = {'check_finite': keyword_73794}
        # Getting the type of 'cho_solve_banded' (line 90)
        cho_solve_banded_73788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'cho_solve_banded', False)
        # Calling cho_solve_banded(args, kwargs) (line 90)
        cho_solve_banded_call_result_73796 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), cho_solve_banded_73788, *[tuple_73789, b_73792], **kwargs_73795)
        
        # Assigning a type to the variable 'x' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'x', cho_solve_banded_call_result_73796)
        
        # Call to assert_array_almost_equal(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'x' (line 91)
        x_73798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_73799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        float_73800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 37), list_73799, float_73800)
        # Adding element type (line 91)
        float_73801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 37), list_73799, float_73801)
        # Adding element type (line 91)
        float_73802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 37), list_73799, float_73802)
        # Adding element type (line 91)
        float_73803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 37), list_73799, float_73803)
        
        # Processing the call keyword arguments (line 91)
        kwargs_73804 = {}
        # Getting the type of 'assert_array_almost_equal' (line 91)
        assert_array_almost_equal_73797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 91)
        assert_array_almost_equal_call_result_73805 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), assert_array_almost_equal_73797, *[x_73798, list_73799], **kwargs_73804)
        
        
        # ################# End of 'test_check_finite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_finite' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_73806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73806)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_finite'
        return stypy_return_type_73806


    @norecursion
    def test_upper_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_upper_real'
        module_type_store = module_type_store.open_function_context('test_upper_real', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_localization', localization)
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_function_name', 'TestCholeskyBanded.test_upper_real')
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholeskyBanded.test_upper_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholeskyBanded.test_upper_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_upper_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_upper_real(...)' code ##################

        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to array(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_73808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_73809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        float_73810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 19), list_73809, float_73810)
        # Adding element type (line 95)
        float_73811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 19), list_73809, float_73811)
        # Adding element type (line 95)
        float_73812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 19), list_73809, float_73812)
        # Adding element type (line 95)
        float_73813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 19), list_73809, float_73813)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 18), list_73808, list_73809)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_73814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        float_73815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 19), list_73814, float_73815)
        # Adding element type (line 96)
        float_73816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 19), list_73814, float_73816)
        # Adding element type (line 96)
        float_73817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 19), list_73814, float_73817)
        # Adding element type (line 96)
        float_73818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 19), list_73814, float_73818)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 18), list_73808, list_73814)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_73819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        float_73820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_73819, float_73820)
        # Adding element type (line 97)
        float_73821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_73819, float_73821)
        # Adding element type (line 97)
        float_73822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_73819, float_73822)
        # Adding element type (line 97)
        float_73823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_73819, float_73823)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 18), list_73808, list_73819)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_73824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        float_73825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_73824, float_73825)
        # Adding element type (line 98)
        float_73826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_73824, float_73826)
        # Adding element type (line 98)
        float_73827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_73824, float_73827)
        # Adding element type (line 98)
        float_73828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_73824, float_73828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 18), list_73808, list_73824)
        
        # Processing the call keyword arguments (line 95)
        kwargs_73829 = {}
        # Getting the type of 'array' (line 95)
        array_73807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'array', False)
        # Calling array(args, kwargs) (line 95)
        array_call_result_73830 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), array_73807, *[list_73808], **kwargs_73829)
        
        # Assigning a type to the variable 'a' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'a', array_call_result_73830)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to array(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_73832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_73833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        float_73834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 20), list_73833, float_73834)
        # Adding element type (line 100)
        float_73835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 20), list_73833, float_73835)
        # Adding element type (line 100)
        float_73836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 20), list_73833, float_73836)
        # Adding element type (line 100)
        float_73837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 20), list_73833, float_73837)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 19), list_73832, list_73833)
        # Adding element type (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_73838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        float_73839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), list_73838, float_73839)
        # Adding element type (line 101)
        float_73840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), list_73838, float_73840)
        # Adding element type (line 101)
        float_73841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), list_73838, float_73841)
        # Adding element type (line 101)
        float_73842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), list_73838, float_73842)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 19), list_73832, list_73838)
        
        # Processing the call keyword arguments (line 100)
        kwargs_73843 = {}
        # Getting the type of 'array' (line 100)
        array_73831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'array', False)
        # Calling array(args, kwargs) (line 100)
        array_call_result_73844 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), array_73831, *[list_73832], **kwargs_73843)
        
        # Assigning a type to the variable 'ab' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'ab', array_call_result_73844)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to cholesky_banded(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'ab' (line 102)
        ab_73846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'ab', False)
        # Processing the call keyword arguments (line 102)
        # Getting the type of 'False' (line 102)
        False_73847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 38), 'False', False)
        keyword_73848 = False_73847
        kwargs_73849 = {'lower': keyword_73848}
        # Getting the type of 'cholesky_banded' (line 102)
        cholesky_banded_73845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'cholesky_banded', False)
        # Calling cholesky_banded(args, kwargs) (line 102)
        cholesky_banded_call_result_73850 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), cholesky_banded_73845, *[ab_73846], **kwargs_73849)
        
        # Assigning a type to the variable 'c' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'c', cholesky_banded_call_result_73850)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to zeros_like(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'a' (line 103)
        a_73852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'a', False)
        # Processing the call keyword arguments (line 103)
        kwargs_73853 = {}
        # Getting the type of 'zeros_like' (line 103)
        zeros_like_73851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'zeros_like', False)
        # Calling zeros_like(args, kwargs) (line 103)
        zeros_like_call_result_73854 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), zeros_like_73851, *[a_73852], **kwargs_73853)
        
        # Assigning a type to the variable 'ufac' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'ufac', zeros_like_call_result_73854)
        
        # Assigning a Subscript to a Subscript (line 104):
        
        # Assigning a Subscript to a Subscript (line 104):
        
        # Obtaining the type of the subscript
        int_73855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 49), 'int')
        # Getting the type of 'c' (line 104)
        c_73856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 47), 'c')
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___73857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 47), c_73856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_73858 = invoke(stypy.reporting.localization.Localization(__file__, 104, 47), getitem___73857, int_73855)
        
        # Getting the type of 'ufac' (line 104)
        ufac_73859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'ufac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_73860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        
        # Call to list(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to range(...): (line 104)
        # Processing the call arguments (line 104)
        int_73863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'int')
        # Processing the call keyword arguments (line 104)
        kwargs_73864 = {}
        # Getting the type of 'range' (line 104)
        range_73862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'range', False)
        # Calling range(args, kwargs) (line 104)
        range_call_result_73865 = invoke(stypy.reporting.localization.Localization(__file__, 104, 18), range_73862, *[int_73863], **kwargs_73864)
        
        # Processing the call keyword arguments (line 104)
        kwargs_73866 = {}
        # Getting the type of 'list' (line 104)
        list_73861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'list', False)
        # Calling list(args, kwargs) (line 104)
        list_call_result_73867 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), list_73861, *[range_call_result_73865], **kwargs_73866)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), tuple_73860, list_call_result_73867)
        # Adding element type (line 104)
        
        # Call to list(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to range(...): (line 104)
        # Processing the call arguments (line 104)
        int_73870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'int')
        # Processing the call keyword arguments (line 104)
        kwargs_73871 = {}
        # Getting the type of 'range' (line 104)
        range_73869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'range', False)
        # Calling range(args, kwargs) (line 104)
        range_call_result_73872 = invoke(stypy.reporting.localization.Localization(__file__, 104, 34), range_73869, *[int_73870], **kwargs_73871)
        
        # Processing the call keyword arguments (line 104)
        kwargs_73873 = {}
        # Getting the type of 'list' (line 104)
        list_73868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'list', False)
        # Calling list(args, kwargs) (line 104)
        list_call_result_73874 = invoke(stypy.reporting.localization.Localization(__file__, 104, 29), list_73868, *[range_call_result_73872], **kwargs_73873)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 13), tuple_73860, list_call_result_73874)
        
        # Storing an element on a container (line 104)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), ufac_73859, (tuple_73860, subscript_call_result_73858))
        
        # Assigning a Subscript to a Subscript (line 105):
        
        # Assigning a Subscript to a Subscript (line 105):
        
        # Obtaining the type of the subscript
        int_73875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 39), 'int')
        int_73876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 42), 'int')
        slice_73877 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 105, 37), int_73876, None, None)
        # Getting the type of 'c' (line 105)
        c_73878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 37), 'c')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___73879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 37), c_73878, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_73880 = invoke(stypy.reporting.localization.Localization(__file__, 105, 37), getitem___73879, (int_73875, slice_73877))
        
        # Getting the type of 'ufac' (line 105)
        ufac_73881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'ufac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_73882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_73883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        int_73884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), tuple_73883, int_73884)
        # Adding element type (line 105)
        int_73885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), tuple_73883, int_73885)
        # Adding element type (line 105)
        int_73886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), tuple_73883, int_73886)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 13), tuple_73882, tuple_73883)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_73887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        int_73888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 25), tuple_73887, int_73888)
        # Adding element type (line 105)
        int_73889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 25), tuple_73887, int_73889)
        # Adding element type (line 105)
        int_73890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 25), tuple_73887, int_73890)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 13), tuple_73882, tuple_73887)
        
        # Storing an element on a container (line 105)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 8), ufac_73881, (tuple_73882, subscript_call_result_73880))
        
        # Call to assert_array_almost_equal(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'a' (line 106)
        a_73892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'a', False)
        
        # Call to dot(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'ufac' (line 106)
        ufac_73894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 41), 'ufac', False)
        # Obtaining the member 'T' of a type (line 106)
        T_73895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 41), ufac_73894, 'T')
        # Getting the type of 'ufac' (line 106)
        ufac_73896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 49), 'ufac', False)
        # Processing the call keyword arguments (line 106)
        kwargs_73897 = {}
        # Getting the type of 'dot' (line 106)
        dot_73893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 37), 'dot', False)
        # Calling dot(args, kwargs) (line 106)
        dot_call_result_73898 = invoke(stypy.reporting.localization.Localization(__file__, 106, 37), dot_73893, *[T_73895, ufac_73896], **kwargs_73897)
        
        # Processing the call keyword arguments (line 106)
        kwargs_73899 = {}
        # Getting the type of 'assert_array_almost_equal' (line 106)
        assert_array_almost_equal_73891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 106)
        assert_array_almost_equal_call_result_73900 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_array_almost_equal_73891, *[a_73892, dot_call_result_73898], **kwargs_73899)
        
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to array(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_73902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        float_73903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 18), list_73902, float_73903)
        # Adding element type (line 108)
        float_73904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 18), list_73902, float_73904)
        # Adding element type (line 108)
        float_73905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 18), list_73902, float_73905)
        # Adding element type (line 108)
        float_73906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 18), list_73902, float_73906)
        
        # Processing the call keyword arguments (line 108)
        kwargs_73907 = {}
        # Getting the type of 'array' (line 108)
        array_73901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'array', False)
        # Calling array(args, kwargs) (line 108)
        array_call_result_73908 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), array_73901, *[list_73902], **kwargs_73907)
        
        # Assigning a type to the variable 'b' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'b', array_call_result_73908)
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to cho_solve_banded(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'tuple' (line 109)
        tuple_73910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 109)
        # Adding element type (line 109)
        # Getting the type of 'c' (line 109)
        c_73911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 30), tuple_73910, c_73911)
        # Adding element type (line 109)
        # Getting the type of 'False' (line 109)
        False_73912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 30), tuple_73910, False_73912)
        
        # Getting the type of 'b' (line 109)
        b_73913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 41), 'b', False)
        # Processing the call keyword arguments (line 109)
        kwargs_73914 = {}
        # Getting the type of 'cho_solve_banded' (line 109)
        cho_solve_banded_73909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'cho_solve_banded', False)
        # Calling cho_solve_banded(args, kwargs) (line 109)
        cho_solve_banded_call_result_73915 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), cho_solve_banded_73909, *[tuple_73910, b_73913], **kwargs_73914)
        
        # Assigning a type to the variable 'x' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'x', cho_solve_banded_call_result_73915)
        
        # Call to assert_array_almost_equal(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'x' (line 110)
        x_73917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_73918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        float_73919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 37), list_73918, float_73919)
        # Adding element type (line 110)
        float_73920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 37), list_73918, float_73920)
        # Adding element type (line 110)
        float_73921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 37), list_73918, float_73921)
        # Adding element type (line 110)
        float_73922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 37), list_73918, float_73922)
        
        # Processing the call keyword arguments (line 110)
        kwargs_73923 = {}
        # Getting the type of 'assert_array_almost_equal' (line 110)
        assert_array_almost_equal_73916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 110)
        assert_array_almost_equal_call_result_73924 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assert_array_almost_equal_73916, *[x_73917, list_73918], **kwargs_73923)
        
        
        # ################# End of 'test_upper_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_upper_real' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_73925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73925)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_upper_real'
        return stypy_return_type_73925


    @norecursion
    def test_upper_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_upper_complex'
        module_type_store = module_type_store.open_function_context('test_upper_complex', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_localization', localization)
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_function_name', 'TestCholeskyBanded.test_upper_complex')
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholeskyBanded.test_upper_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholeskyBanded.test_upper_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_upper_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_upper_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to array(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_73927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_73928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        float_73929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 19), list_73928, float_73929)
        # Adding element type (line 114)
        float_73930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 19), list_73928, float_73930)
        # Adding element type (line 114)
        float_73931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 19), list_73928, float_73931)
        # Adding element type (line 114)
        float_73932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 19), list_73928, float_73932)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 18), list_73927, list_73928)
        # Adding element type (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_73933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        float_73934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_73933, float_73934)
        # Adding element type (line 115)
        float_73935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_73933, float_73935)
        # Adding element type (line 115)
        float_73936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_73933, float_73936)
        # Adding element type (line 115)
        float_73937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_73933, float_73937)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 18), list_73927, list_73933)
        # Adding element type (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_73938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        float_73939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_73938, float_73939)
        # Adding element type (line 116)
        float_73940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_73938, float_73940)
        # Adding element type (line 116)
        float_73941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_73938, float_73941)
        # Adding element type (line 116)
        complex_73942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 35), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_73938, complex_73942)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 18), list_73927, list_73938)
        # Adding element type (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_73943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        float_73944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 19), list_73943, float_73944)
        # Adding element type (line 117)
        float_73945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 19), list_73943, float_73945)
        # Adding element type (line 117)
        complex_73946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 30), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 19), list_73943, complex_73946)
        # Adding element type (line 117)
        float_73947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 19), list_73943, float_73947)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 18), list_73927, list_73943)
        
        # Processing the call keyword arguments (line 114)
        kwargs_73948 = {}
        # Getting the type of 'array' (line 114)
        array_73926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'array', False)
        # Calling array(args, kwargs) (line 114)
        array_call_result_73949 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), array_73926, *[list_73927], **kwargs_73948)
        
        # Assigning a type to the variable 'a' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'a', array_call_result_73949)
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to array(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_73951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_73952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        float_73953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), list_73952, float_73953)
        # Adding element type (line 119)
        float_73954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), list_73952, float_73954)
        # Adding element type (line 119)
        float_73955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), list_73952, float_73955)
        # Adding element type (line 119)
        complex_73956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 37), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), list_73952, complex_73956)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 19), list_73951, list_73952)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_73957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        float_73958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 20), list_73957, float_73958)
        # Adding element type (line 120)
        float_73959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 20), list_73957, float_73959)
        # Adding element type (line 120)
        float_73960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 20), list_73957, float_73960)
        # Adding element type (line 120)
        float_73961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 20), list_73957, float_73961)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 19), list_73951, list_73957)
        
        # Processing the call keyword arguments (line 119)
        kwargs_73962 = {}
        # Getting the type of 'array' (line 119)
        array_73950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 13), 'array', False)
        # Calling array(args, kwargs) (line 119)
        array_call_result_73963 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), array_73950, *[list_73951], **kwargs_73962)
        
        # Assigning a type to the variable 'ab' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'ab', array_call_result_73963)
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to cholesky_banded(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'ab' (line 121)
        ab_73965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'ab', False)
        # Processing the call keyword arguments (line 121)
        # Getting the type of 'False' (line 121)
        False_73966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 38), 'False', False)
        keyword_73967 = False_73966
        kwargs_73968 = {'lower': keyword_73967}
        # Getting the type of 'cholesky_banded' (line 121)
        cholesky_banded_73964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'cholesky_banded', False)
        # Calling cholesky_banded(args, kwargs) (line 121)
        cholesky_banded_call_result_73969 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), cholesky_banded_73964, *[ab_73965], **kwargs_73968)
        
        # Assigning a type to the variable 'c' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'c', cholesky_banded_call_result_73969)
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to zeros_like(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'a' (line 122)
        a_73971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 26), 'a', False)
        # Processing the call keyword arguments (line 122)
        kwargs_73972 = {}
        # Getting the type of 'zeros_like' (line 122)
        zeros_like_73970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'zeros_like', False)
        # Calling zeros_like(args, kwargs) (line 122)
        zeros_like_call_result_73973 = invoke(stypy.reporting.localization.Localization(__file__, 122, 15), zeros_like_73970, *[a_73971], **kwargs_73972)
        
        # Assigning a type to the variable 'ufac' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'ufac', zeros_like_call_result_73973)
        
        # Assigning a Subscript to a Subscript (line 123):
        
        # Assigning a Subscript to a Subscript (line 123):
        
        # Obtaining the type of the subscript
        int_73974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'int')
        # Getting the type of 'c' (line 123)
        c_73975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'c')
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___73976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 47), c_73975, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_73977 = invoke(stypy.reporting.localization.Localization(__file__, 123, 47), getitem___73976, int_73974)
        
        # Getting the type of 'ufac' (line 123)
        ufac_73978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'ufac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_73979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        
        # Call to list(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to range(...): (line 123)
        # Processing the call arguments (line 123)
        int_73982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 24), 'int')
        # Processing the call keyword arguments (line 123)
        kwargs_73983 = {}
        # Getting the type of 'range' (line 123)
        range_73981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'range', False)
        # Calling range(args, kwargs) (line 123)
        range_call_result_73984 = invoke(stypy.reporting.localization.Localization(__file__, 123, 18), range_73981, *[int_73982], **kwargs_73983)
        
        # Processing the call keyword arguments (line 123)
        kwargs_73985 = {}
        # Getting the type of 'list' (line 123)
        list_73980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 13), 'list', False)
        # Calling list(args, kwargs) (line 123)
        list_call_result_73986 = invoke(stypy.reporting.localization.Localization(__file__, 123, 13), list_73980, *[range_call_result_73984], **kwargs_73985)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 13), tuple_73979, list_call_result_73986)
        # Adding element type (line 123)
        
        # Call to list(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to range(...): (line 123)
        # Processing the call arguments (line 123)
        int_73989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 40), 'int')
        # Processing the call keyword arguments (line 123)
        kwargs_73990 = {}
        # Getting the type of 'range' (line 123)
        range_73988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 34), 'range', False)
        # Calling range(args, kwargs) (line 123)
        range_call_result_73991 = invoke(stypy.reporting.localization.Localization(__file__, 123, 34), range_73988, *[int_73989], **kwargs_73990)
        
        # Processing the call keyword arguments (line 123)
        kwargs_73992 = {}
        # Getting the type of 'list' (line 123)
        list_73987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'list', False)
        # Calling list(args, kwargs) (line 123)
        list_call_result_73993 = invoke(stypy.reporting.localization.Localization(__file__, 123, 29), list_73987, *[range_call_result_73991], **kwargs_73992)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 13), tuple_73979, list_call_result_73993)
        
        # Storing an element on a container (line 123)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), ufac_73978, (tuple_73979, subscript_call_result_73977))
        
        # Assigning a Subscript to a Subscript (line 124):
        
        # Assigning a Subscript to a Subscript (line 124):
        
        # Obtaining the type of the subscript
        int_73994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 39), 'int')
        int_73995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'int')
        slice_73996 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 124, 37), int_73995, None, None)
        # Getting the type of 'c' (line 124)
        c_73997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'c')
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___73998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 37), c_73997, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_73999 = invoke(stypy.reporting.localization.Localization(__file__, 124, 37), getitem___73998, (int_73994, slice_73996))
        
        # Getting the type of 'ufac' (line 124)
        ufac_74000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'ufac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_74001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_74002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        int_74003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 14), tuple_74002, int_74003)
        # Adding element type (line 124)
        int_74004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 14), tuple_74002, int_74004)
        # Adding element type (line 124)
        int_74005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 14), tuple_74002, int_74005)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 13), tuple_74001, tuple_74002)
        # Adding element type (line 124)
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_74006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        int_74007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 25), tuple_74006, int_74007)
        # Adding element type (line 124)
        int_74008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 25), tuple_74006, int_74008)
        # Adding element type (line 124)
        int_74009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 25), tuple_74006, int_74009)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 13), tuple_74001, tuple_74006)
        
        # Storing an element on a container (line 124)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), ufac_74000, (tuple_74001, subscript_call_result_73999))
        
        # Call to assert_array_almost_equal(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'a' (line 125)
        a_74011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'a', False)
        
        # Call to dot(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to conj(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_74015 = {}
        # Getting the type of 'ufac' (line 125)
        ufac_74013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'ufac', False)
        # Obtaining the member 'conj' of a type (line 125)
        conj_74014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 41), ufac_74013, 'conj')
        # Calling conj(args, kwargs) (line 125)
        conj_call_result_74016 = invoke(stypy.reporting.localization.Localization(__file__, 125, 41), conj_74014, *[], **kwargs_74015)
        
        # Obtaining the member 'T' of a type (line 125)
        T_74017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 41), conj_call_result_74016, 'T')
        # Getting the type of 'ufac' (line 125)
        ufac_74018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 56), 'ufac', False)
        # Processing the call keyword arguments (line 125)
        kwargs_74019 = {}
        # Getting the type of 'dot' (line 125)
        dot_74012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'dot', False)
        # Calling dot(args, kwargs) (line 125)
        dot_call_result_74020 = invoke(stypy.reporting.localization.Localization(__file__, 125, 37), dot_74012, *[T_74017, ufac_74018], **kwargs_74019)
        
        # Processing the call keyword arguments (line 125)
        kwargs_74021 = {}
        # Getting the type of 'assert_array_almost_equal' (line 125)
        assert_array_almost_equal_74010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 125)
        assert_array_almost_equal_call_result_74022 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assert_array_almost_equal_74010, *[a_74011, dot_call_result_74020], **kwargs_74021)
        
        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to array(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_74024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        float_74025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 18), list_74024, float_74025)
        # Adding element type (line 127)
        float_74026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 18), list_74024, float_74026)
        # Adding element type (line 127)
        float_74027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'float')
        complex_74028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 33), 'complex')
        # Applying the binary operator '-' (line 127)
        result_sub_74029 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 29), '-', float_74027, complex_74028)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 18), list_74024, result_sub_74029)
        # Adding element type (line 127)
        complex_74030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 39), 'complex')
        float_74031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 46), 'float')
        # Applying the binary operator '+' (line 127)
        result_add_74032 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 39), '+', complex_74030, float_74031)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 18), list_74024, result_add_74032)
        
        # Processing the call keyword arguments (line 127)
        kwargs_74033 = {}
        # Getting the type of 'array' (line 127)
        array_74023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'array', False)
        # Calling array(args, kwargs) (line 127)
        array_call_result_74034 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), array_74023, *[list_74024], **kwargs_74033)
        
        # Assigning a type to the variable 'b' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'b', array_call_result_74034)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to cho_solve_banded(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'tuple' (line 128)
        tuple_74036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'c' (line 128)
        c_74037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 30), tuple_74036, c_74037)
        # Adding element type (line 128)
        # Getting the type of 'False' (line 128)
        False_74038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 33), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 30), tuple_74036, False_74038)
        
        # Getting the type of 'b' (line 128)
        b_74039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'b', False)
        # Processing the call keyword arguments (line 128)
        kwargs_74040 = {}
        # Getting the type of 'cho_solve_banded' (line 128)
        cho_solve_banded_74035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'cho_solve_banded', False)
        # Calling cho_solve_banded(args, kwargs) (line 128)
        cho_solve_banded_call_result_74041 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), cho_solve_banded_74035, *[tuple_74036, b_74039], **kwargs_74040)
        
        # Assigning a type to the variable 'x' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'x', cho_solve_banded_call_result_74041)
        
        # Call to assert_array_almost_equal(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'x' (line 129)
        x_74043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_74044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        float_74045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 37), list_74044, float_74045)
        # Adding element type (line 129)
        float_74046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 37), list_74044, float_74046)
        # Adding element type (line 129)
        float_74047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 37), list_74044, float_74047)
        # Adding element type (line 129)
        float_74048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 37), list_74044, float_74048)
        
        # Processing the call keyword arguments (line 129)
        kwargs_74049 = {}
        # Getting the type of 'assert_array_almost_equal' (line 129)
        assert_array_almost_equal_74042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 129)
        assert_array_almost_equal_call_result_74050 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assert_array_almost_equal_74042, *[x_74043, list_74044], **kwargs_74049)
        
        
        # ################# End of 'test_upper_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_upper_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_74051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74051)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_upper_complex'
        return stypy_return_type_74051


    @norecursion
    def test_lower_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lower_real'
        module_type_store = module_type_store.open_function_context('test_lower_real', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_localization', localization)
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_function_name', 'TestCholeskyBanded.test_lower_real')
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholeskyBanded.test_lower_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholeskyBanded.test_lower_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lower_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lower_real(...)' code ##################

        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to array(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_74053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_74054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        float_74055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), list_74054, float_74055)
        # Adding element type (line 133)
        float_74056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), list_74054, float_74056)
        # Adding element type (line 133)
        float_74057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), list_74054, float_74057)
        # Adding element type (line 133)
        float_74058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), list_74054, float_74058)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 18), list_74053, list_74054)
        # Adding element type (line 133)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_74059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        float_74060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), list_74059, float_74060)
        # Adding element type (line 134)
        float_74061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), list_74059, float_74061)
        # Adding element type (line 134)
        float_74062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), list_74059, float_74062)
        # Adding element type (line 134)
        float_74063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), list_74059, float_74063)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 18), list_74053, list_74059)
        # Adding element type (line 133)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_74064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        float_74065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), list_74064, float_74065)
        # Adding element type (line 135)
        float_74066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), list_74064, float_74066)
        # Adding element type (line 135)
        float_74067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), list_74064, float_74067)
        # Adding element type (line 135)
        float_74068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), list_74064, float_74068)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 18), list_74053, list_74064)
        # Adding element type (line 133)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_74069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        float_74070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_74069, float_74070)
        # Adding element type (line 136)
        float_74071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_74069, float_74071)
        # Adding element type (line 136)
        float_74072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_74069, float_74072)
        # Adding element type (line 136)
        float_74073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_74069, float_74073)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 18), list_74053, list_74069)
        
        # Processing the call keyword arguments (line 133)
        kwargs_74074 = {}
        # Getting the type of 'array' (line 133)
        array_74052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'array', False)
        # Calling array(args, kwargs) (line 133)
        array_call_result_74075 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), array_74052, *[list_74053], **kwargs_74074)
        
        # Assigning a type to the variable 'a' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'a', array_call_result_74075)
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to array(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_74077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_74078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        float_74079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_74078, float_74079)
        # Adding element type (line 138)
        float_74080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_74078, float_74080)
        # Adding element type (line 138)
        float_74081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_74078, float_74081)
        # Adding element type (line 138)
        float_74082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_74078, float_74082)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), list_74077, list_74078)
        # Adding element type (line 138)
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_74083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        float_74084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 20), list_74083, float_74084)
        # Adding element type (line 139)
        float_74085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 20), list_74083, float_74085)
        # Adding element type (line 139)
        float_74086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 20), list_74083, float_74086)
        # Adding element type (line 139)
        float_74087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 20), list_74083, float_74087)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), list_74077, list_74083)
        
        # Processing the call keyword arguments (line 138)
        kwargs_74088 = {}
        # Getting the type of 'array' (line 138)
        array_74076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'array', False)
        # Calling array(args, kwargs) (line 138)
        array_call_result_74089 = invoke(stypy.reporting.localization.Localization(__file__, 138, 13), array_74076, *[list_74077], **kwargs_74088)
        
        # Assigning a type to the variable 'ab' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'ab', array_call_result_74089)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to cholesky_banded(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'ab' (line 140)
        ab_74091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'ab', False)
        # Processing the call keyword arguments (line 140)
        # Getting the type of 'True' (line 140)
        True_74092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'True', False)
        keyword_74093 = True_74092
        kwargs_74094 = {'lower': keyword_74093}
        # Getting the type of 'cholesky_banded' (line 140)
        cholesky_banded_74090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'cholesky_banded', False)
        # Calling cholesky_banded(args, kwargs) (line 140)
        cholesky_banded_call_result_74095 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), cholesky_banded_74090, *[ab_74091], **kwargs_74094)
        
        # Assigning a type to the variable 'c' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'c', cholesky_banded_call_result_74095)
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to zeros_like(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'a' (line 141)
        a_74097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'a', False)
        # Processing the call keyword arguments (line 141)
        kwargs_74098 = {}
        # Getting the type of 'zeros_like' (line 141)
        zeros_like_74096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'zeros_like', False)
        # Calling zeros_like(args, kwargs) (line 141)
        zeros_like_call_result_74099 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), zeros_like_74096, *[a_74097], **kwargs_74098)
        
        # Assigning a type to the variable 'lfac' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'lfac', zeros_like_call_result_74099)
        
        # Assigning a Subscript to a Subscript (line 142):
        
        # Assigning a Subscript to a Subscript (line 142):
        
        # Obtaining the type of the subscript
        int_74100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 49), 'int')
        # Getting the type of 'c' (line 142)
        c_74101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 47), 'c')
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___74102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 47), c_74101, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_74103 = invoke(stypy.reporting.localization.Localization(__file__, 142, 47), getitem___74102, int_74100)
        
        # Getting the type of 'lfac' (line 142)
        lfac_74104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'lfac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_74105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        
        # Call to list(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to range(...): (line 142)
        # Processing the call arguments (line 142)
        int_74108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 24), 'int')
        # Processing the call keyword arguments (line 142)
        kwargs_74109 = {}
        # Getting the type of 'range' (line 142)
        range_74107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'range', False)
        # Calling range(args, kwargs) (line 142)
        range_call_result_74110 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), range_74107, *[int_74108], **kwargs_74109)
        
        # Processing the call keyword arguments (line 142)
        kwargs_74111 = {}
        # Getting the type of 'list' (line 142)
        list_74106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'list', False)
        # Calling list(args, kwargs) (line 142)
        list_call_result_74112 = invoke(stypy.reporting.localization.Localization(__file__, 142, 13), list_74106, *[range_call_result_74110], **kwargs_74111)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 13), tuple_74105, list_call_result_74112)
        # Adding element type (line 142)
        
        # Call to list(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to range(...): (line 142)
        # Processing the call arguments (line 142)
        int_74115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'int')
        # Processing the call keyword arguments (line 142)
        kwargs_74116 = {}
        # Getting the type of 'range' (line 142)
        range_74114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'range', False)
        # Calling range(args, kwargs) (line 142)
        range_call_result_74117 = invoke(stypy.reporting.localization.Localization(__file__, 142, 34), range_74114, *[int_74115], **kwargs_74116)
        
        # Processing the call keyword arguments (line 142)
        kwargs_74118 = {}
        # Getting the type of 'list' (line 142)
        list_74113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'list', False)
        # Calling list(args, kwargs) (line 142)
        list_call_result_74119 = invoke(stypy.reporting.localization.Localization(__file__, 142, 29), list_74113, *[range_call_result_74117], **kwargs_74118)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 13), tuple_74105, list_call_result_74119)
        
        # Storing an element on a container (line 142)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 8), lfac_74104, (tuple_74105, subscript_call_result_74103))
        
        # Assigning a Subscript to a Subscript (line 143):
        
        # Assigning a Subscript to a Subscript (line 143):
        
        # Obtaining the type of the subscript
        int_74120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 39), 'int')
        int_74121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 43), 'int')
        slice_74122 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 143, 37), None, int_74121, None)
        # Getting the type of 'c' (line 143)
        c_74123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 37), 'c')
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___74124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 37), c_74123, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_74125 = invoke(stypy.reporting.localization.Localization(__file__, 143, 37), getitem___74124, (int_74120, slice_74122))
        
        # Getting the type of 'lfac' (line 143)
        lfac_74126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'lfac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 143)
        tuple_74127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 143)
        # Adding element type (line 143)
        
        # Obtaining an instance of the builtin type 'tuple' (line 143)
        tuple_74128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 143)
        # Adding element type (line 143)
        int_74129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 14), tuple_74128, int_74129)
        # Adding element type (line 143)
        int_74130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 14), tuple_74128, int_74130)
        # Adding element type (line 143)
        int_74131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 14), tuple_74128, int_74131)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 13), tuple_74127, tuple_74128)
        # Adding element type (line 143)
        
        # Obtaining an instance of the builtin type 'tuple' (line 143)
        tuple_74132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 143)
        # Adding element type (line 143)
        int_74133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 25), tuple_74132, int_74133)
        # Adding element type (line 143)
        int_74134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 25), tuple_74132, int_74134)
        # Adding element type (line 143)
        int_74135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 25), tuple_74132, int_74135)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 13), tuple_74127, tuple_74132)
        
        # Storing an element on a container (line 143)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 8), lfac_74126, (tuple_74127, subscript_call_result_74125))
        
        # Call to assert_array_almost_equal(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'a' (line 144)
        a_74137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'a', False)
        
        # Call to dot(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'lfac' (line 144)
        lfac_74139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'lfac', False)
        # Getting the type of 'lfac' (line 144)
        lfac_74140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 47), 'lfac', False)
        # Obtaining the member 'T' of a type (line 144)
        T_74141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 47), lfac_74140, 'T')
        # Processing the call keyword arguments (line 144)
        kwargs_74142 = {}
        # Getting the type of 'dot' (line 144)
        dot_74138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'dot', False)
        # Calling dot(args, kwargs) (line 144)
        dot_call_result_74143 = invoke(stypy.reporting.localization.Localization(__file__, 144, 37), dot_74138, *[lfac_74139, T_74141], **kwargs_74142)
        
        # Processing the call keyword arguments (line 144)
        kwargs_74144 = {}
        # Getting the type of 'assert_array_almost_equal' (line 144)
        assert_array_almost_equal_74136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 144)
        assert_array_almost_equal_call_result_74145 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assert_array_almost_equal_74136, *[a_74137, dot_call_result_74143], **kwargs_74144)
        
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to array(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'list' (line 146)
        list_74147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 146)
        # Adding element type (line 146)
        float_74148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 18), list_74147, float_74148)
        # Adding element type (line 146)
        float_74149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 18), list_74147, float_74149)
        # Adding element type (line 146)
        float_74150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 18), list_74147, float_74150)
        # Adding element type (line 146)
        float_74151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 18), list_74147, float_74151)
        
        # Processing the call keyword arguments (line 146)
        kwargs_74152 = {}
        # Getting the type of 'array' (line 146)
        array_74146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'array', False)
        # Calling array(args, kwargs) (line 146)
        array_call_result_74153 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), array_74146, *[list_74147], **kwargs_74152)
        
        # Assigning a type to the variable 'b' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'b', array_call_result_74153)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to cho_solve_banded(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_74155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        # Getting the type of 'c' (line 147)
        c_74156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 30), tuple_74155, c_74156)
        # Adding element type (line 147)
        # Getting the type of 'True' (line 147)
        True_74157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 30), tuple_74155, True_74157)
        
        # Getting the type of 'b' (line 147)
        b_74158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 40), 'b', False)
        # Processing the call keyword arguments (line 147)
        kwargs_74159 = {}
        # Getting the type of 'cho_solve_banded' (line 147)
        cho_solve_banded_74154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'cho_solve_banded', False)
        # Calling cho_solve_banded(args, kwargs) (line 147)
        cho_solve_banded_call_result_74160 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), cho_solve_banded_74154, *[tuple_74155, b_74158], **kwargs_74159)
        
        # Assigning a type to the variable 'x' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'x', cho_solve_banded_call_result_74160)
        
        # Call to assert_array_almost_equal(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'x' (line 148)
        x_74162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_74163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        float_74164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 37), list_74163, float_74164)
        # Adding element type (line 148)
        float_74165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 37), list_74163, float_74165)
        # Adding element type (line 148)
        float_74166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 37), list_74163, float_74166)
        # Adding element type (line 148)
        float_74167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 37), list_74163, float_74167)
        
        # Processing the call keyword arguments (line 148)
        kwargs_74168 = {}
        # Getting the type of 'assert_array_almost_equal' (line 148)
        assert_array_almost_equal_74161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 148)
        assert_array_almost_equal_call_result_74169 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), assert_array_almost_equal_74161, *[x_74162, list_74163], **kwargs_74168)
        
        
        # ################# End of 'test_lower_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lower_real' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_74170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74170)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lower_real'
        return stypy_return_type_74170


    @norecursion
    def test_lower_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lower_complex'
        module_type_store = module_type_store.open_function_context('test_lower_complex', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_localization', localization)
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_function_name', 'TestCholeskyBanded.test_lower_complex')
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCholeskyBanded.test_lower_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholeskyBanded.test_lower_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lower_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lower_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to array(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_74172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_74173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        float_74174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_74173, float_74174)
        # Adding element type (line 152)
        float_74175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_74173, float_74175)
        # Adding element type (line 152)
        float_74176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_74173, float_74176)
        # Adding element type (line 152)
        float_74177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_74173, float_74177)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 18), list_74172, list_74173)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_74178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        # Adding element type (line 153)
        float_74179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_74178, float_74179)
        # Adding element type (line 153)
        float_74180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_74178, float_74180)
        # Adding element type (line 153)
        float_74181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_74178, float_74181)
        # Adding element type (line 153)
        float_74182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_74178, float_74182)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 18), list_74172, list_74178)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_74183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        float_74184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_74183, float_74184)
        # Adding element type (line 154)
        float_74185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_74183, float_74185)
        # Adding element type (line 154)
        float_74186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_74183, float_74186)
        # Adding element type (line 154)
        complex_74187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 35), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_74183, complex_74187)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 18), list_74172, list_74183)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_74188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        # Adding element type (line 155)
        float_74189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_74188, float_74189)
        # Adding element type (line 155)
        float_74190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_74188, float_74190)
        # Adding element type (line 155)
        complex_74191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 30), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_74188, complex_74191)
        # Adding element type (line 155)
        float_74192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_74188, float_74192)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 18), list_74172, list_74188)
        
        # Processing the call keyword arguments (line 152)
        kwargs_74193 = {}
        # Getting the type of 'array' (line 152)
        array_74171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'array', False)
        # Calling array(args, kwargs) (line 152)
        array_call_result_74194 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), array_74171, *[list_74172], **kwargs_74193)
        
        # Assigning a type to the variable 'a' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'a', array_call_result_74194)
        
        # Assigning a Call to a Name (line 157):
        
        # Assigning a Call to a Name (line 157):
        
        # Call to array(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_74196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        # Adding element type (line 157)
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_74197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        # Adding element type (line 157)
        float_74198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), list_74197, float_74198)
        # Adding element type (line 157)
        float_74199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), list_74197, float_74199)
        # Adding element type (line 157)
        float_74200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), list_74197, float_74200)
        # Adding element type (line 157)
        float_74201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), list_74197, float_74201)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 19), list_74196, list_74197)
        # Adding element type (line 157)
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_74202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        float_74203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 20), list_74202, float_74203)
        # Adding element type (line 158)
        float_74204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 20), list_74202, float_74204)
        # Adding element type (line 158)
        complex_74205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 31), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 20), list_74202, complex_74205)
        # Adding element type (line 158)
        float_74206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 20), list_74202, float_74206)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 19), list_74196, list_74202)
        
        # Processing the call keyword arguments (line 157)
        kwargs_74207 = {}
        # Getting the type of 'array' (line 157)
        array_74195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'array', False)
        # Calling array(args, kwargs) (line 157)
        array_call_result_74208 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), array_74195, *[list_74196], **kwargs_74207)
        
        # Assigning a type to the variable 'ab' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'ab', array_call_result_74208)
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to cholesky_banded(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'ab' (line 159)
        ab_74210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'ab', False)
        # Processing the call keyword arguments (line 159)
        # Getting the type of 'True' (line 159)
        True_74211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 38), 'True', False)
        keyword_74212 = True_74211
        kwargs_74213 = {'lower': keyword_74212}
        # Getting the type of 'cholesky_banded' (line 159)
        cholesky_banded_74209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'cholesky_banded', False)
        # Calling cholesky_banded(args, kwargs) (line 159)
        cholesky_banded_call_result_74214 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), cholesky_banded_74209, *[ab_74210], **kwargs_74213)
        
        # Assigning a type to the variable 'c' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'c', cholesky_banded_call_result_74214)
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to zeros_like(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'a' (line 160)
        a_74216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'a', False)
        # Processing the call keyword arguments (line 160)
        kwargs_74217 = {}
        # Getting the type of 'zeros_like' (line 160)
        zeros_like_74215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'zeros_like', False)
        # Calling zeros_like(args, kwargs) (line 160)
        zeros_like_call_result_74218 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), zeros_like_74215, *[a_74216], **kwargs_74217)
        
        # Assigning a type to the variable 'lfac' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'lfac', zeros_like_call_result_74218)
        
        # Assigning a Subscript to a Subscript (line 161):
        
        # Assigning a Subscript to a Subscript (line 161):
        
        # Obtaining the type of the subscript
        int_74219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 49), 'int')
        # Getting the type of 'c' (line 161)
        c_74220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 47), 'c')
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___74221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 47), c_74220, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_74222 = invoke(stypy.reporting.localization.Localization(__file__, 161, 47), getitem___74221, int_74219)
        
        # Getting the type of 'lfac' (line 161)
        lfac_74223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'lfac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_74224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        
        # Call to list(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Call to range(...): (line 161)
        # Processing the call arguments (line 161)
        int_74227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 24), 'int')
        # Processing the call keyword arguments (line 161)
        kwargs_74228 = {}
        # Getting the type of 'range' (line 161)
        range_74226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), 'range', False)
        # Calling range(args, kwargs) (line 161)
        range_call_result_74229 = invoke(stypy.reporting.localization.Localization(__file__, 161, 18), range_74226, *[int_74227], **kwargs_74228)
        
        # Processing the call keyword arguments (line 161)
        kwargs_74230 = {}
        # Getting the type of 'list' (line 161)
        list_74225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'list', False)
        # Calling list(args, kwargs) (line 161)
        list_call_result_74231 = invoke(stypy.reporting.localization.Localization(__file__, 161, 13), list_74225, *[range_call_result_74229], **kwargs_74230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 13), tuple_74224, list_call_result_74231)
        # Adding element type (line 161)
        
        # Call to list(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Call to range(...): (line 161)
        # Processing the call arguments (line 161)
        int_74234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 40), 'int')
        # Processing the call keyword arguments (line 161)
        kwargs_74235 = {}
        # Getting the type of 'range' (line 161)
        range_74233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'range', False)
        # Calling range(args, kwargs) (line 161)
        range_call_result_74236 = invoke(stypy.reporting.localization.Localization(__file__, 161, 34), range_74233, *[int_74234], **kwargs_74235)
        
        # Processing the call keyword arguments (line 161)
        kwargs_74237 = {}
        # Getting the type of 'list' (line 161)
        list_74232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'list', False)
        # Calling list(args, kwargs) (line 161)
        list_call_result_74238 = invoke(stypy.reporting.localization.Localization(__file__, 161, 29), list_74232, *[range_call_result_74236], **kwargs_74237)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 13), tuple_74224, list_call_result_74238)
        
        # Storing an element on a container (line 161)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 8), lfac_74223, (tuple_74224, subscript_call_result_74222))
        
        # Assigning a Subscript to a Subscript (line 162):
        
        # Assigning a Subscript to a Subscript (line 162):
        
        # Obtaining the type of the subscript
        int_74239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 39), 'int')
        int_74240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 43), 'int')
        slice_74241 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 162, 37), None, int_74240, None)
        # Getting the type of 'c' (line 162)
        c_74242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'c')
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___74243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 37), c_74242, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_74244 = invoke(stypy.reporting.localization.Localization(__file__, 162, 37), getitem___74243, (int_74239, slice_74241))
        
        # Getting the type of 'lfac' (line 162)
        lfac_74245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'lfac')
        
        # Obtaining an instance of the builtin type 'tuple' (line 162)
        tuple_74246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 162)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 162)
        tuple_74247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 162)
        # Adding element type (line 162)
        int_74248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), tuple_74247, int_74248)
        # Adding element type (line 162)
        int_74249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), tuple_74247, int_74249)
        # Adding element type (line 162)
        int_74250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), tuple_74247, int_74250)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 13), tuple_74246, tuple_74247)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 162)
        tuple_74251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 162)
        # Adding element type (line 162)
        int_74252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 25), tuple_74251, int_74252)
        # Adding element type (line 162)
        int_74253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 25), tuple_74251, int_74253)
        # Adding element type (line 162)
        int_74254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 25), tuple_74251, int_74254)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 13), tuple_74246, tuple_74251)
        
        # Storing an element on a container (line 162)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), lfac_74245, (tuple_74246, subscript_call_result_74244))
        
        # Call to assert_array_almost_equal(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'a' (line 163)
        a_74256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'a', False)
        
        # Call to dot(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'lfac' (line 163)
        lfac_74258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'lfac', False)
        
        # Call to conj(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_74261 = {}
        # Getting the type of 'lfac' (line 163)
        lfac_74259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 47), 'lfac', False)
        # Obtaining the member 'conj' of a type (line 163)
        conj_74260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 47), lfac_74259, 'conj')
        # Calling conj(args, kwargs) (line 163)
        conj_call_result_74262 = invoke(stypy.reporting.localization.Localization(__file__, 163, 47), conj_74260, *[], **kwargs_74261)
        
        # Obtaining the member 'T' of a type (line 163)
        T_74263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 47), conj_call_result_74262, 'T')
        # Processing the call keyword arguments (line 163)
        kwargs_74264 = {}
        # Getting the type of 'dot' (line 163)
        dot_74257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'dot', False)
        # Calling dot(args, kwargs) (line 163)
        dot_call_result_74265 = invoke(stypy.reporting.localization.Localization(__file__, 163, 37), dot_74257, *[lfac_74258, T_74263], **kwargs_74264)
        
        # Processing the call keyword arguments (line 163)
        kwargs_74266 = {}
        # Getting the type of 'assert_array_almost_equal' (line 163)
        assert_array_almost_equal_74255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 163)
        assert_array_almost_equal_call_result_74267 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), assert_array_almost_equal_74255, *[a_74256, dot_call_result_74265], **kwargs_74266)
        
        
        # Assigning a Call to a Name (line 165):
        
        # Assigning a Call to a Name (line 165):
        
        # Call to array(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_74269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        float_74270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 18), list_74269, float_74270)
        # Adding element type (line 165)
        complex_74271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 24), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 18), list_74269, complex_74271)
        # Adding element type (line 165)
        complex_74272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 30), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 18), list_74269, complex_74272)
        # Adding element type (line 165)
        float_74273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 18), list_74269, float_74273)
        
        # Processing the call keyword arguments (line 165)
        kwargs_74274 = {}
        # Getting the type of 'array' (line 165)
        array_74268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'array', False)
        # Calling array(args, kwargs) (line 165)
        array_call_result_74275 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), array_74268, *[list_74269], **kwargs_74274)
        
        # Assigning a type to the variable 'b' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'b', array_call_result_74275)
        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to cho_solve_banded(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Obtaining an instance of the builtin type 'tuple' (line 166)
        tuple_74277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 166)
        # Adding element type (line 166)
        # Getting the type of 'c' (line 166)
        c_74278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 30), tuple_74277, c_74278)
        # Adding element type (line 166)
        # Getting the type of 'True' (line 166)
        True_74279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 30), tuple_74277, True_74279)
        
        # Getting the type of 'b' (line 166)
        b_74280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 40), 'b', False)
        # Processing the call keyword arguments (line 166)
        kwargs_74281 = {}
        # Getting the type of 'cho_solve_banded' (line 166)
        cho_solve_banded_74276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'cho_solve_banded', False)
        # Calling cho_solve_banded(args, kwargs) (line 166)
        cho_solve_banded_call_result_74282 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), cho_solve_banded_74276, *[tuple_74277, b_74280], **kwargs_74281)
        
        # Assigning a type to the variable 'x' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'x', cho_solve_banded_call_result_74282)
        
        # Call to assert_array_almost_equal(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'x' (line 167)
        x_74284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 167)
        list_74285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 167)
        # Adding element type (line 167)
        float_74286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 37), list_74285, float_74286)
        # Adding element type (line 167)
        float_74287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 37), list_74285, float_74287)
        # Adding element type (line 167)
        complex_74288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 48), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 37), list_74285, complex_74288)
        # Adding element type (line 167)
        float_74289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 37), list_74285, float_74289)
        
        # Processing the call keyword arguments (line 167)
        kwargs_74290 = {}
        # Getting the type of 'assert_array_almost_equal' (line 167)
        assert_array_almost_equal_74283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 167)
        assert_array_almost_equal_call_result_74291 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), assert_array_almost_equal_74283, *[x_74284, list_74285], **kwargs_74290)
        
        
        # ################# End of 'test_lower_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lower_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_74292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lower_complex'
        return stypy_return_type_74292


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCholeskyBanded.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCholeskyBanded' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'TestCholeskyBanded', TestCholeskyBanded)
# Declaration of the 'TestOverwrite' class

class TestOverwrite(object, ):

    @norecursion
    def test_cholesky(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cholesky'
        module_type_store = module_type_store.open_function_context('test_cholesky', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_cholesky')
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_cholesky.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_cholesky', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cholesky', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cholesky(...)' code ##################

        
        # Call to assert_no_overwrite(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'cholesky' (line 172)
        cholesky_74294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'cholesky', False)
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_74295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        
        # Obtaining an instance of the builtin type 'tuple' (line 172)
        tuple_74296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 172)
        # Adding element type (line 172)
        int_74297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 40), tuple_74296, int_74297)
        # Adding element type (line 172)
        int_74298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 40), tuple_74296, int_74298)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 38), list_74295, tuple_74296)
        
        # Processing the call keyword arguments (line 172)
        kwargs_74299 = {}
        # Getting the type of 'assert_no_overwrite' (line 172)
        assert_no_overwrite_74293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'assert_no_overwrite', False)
        # Calling assert_no_overwrite(args, kwargs) (line 172)
        assert_no_overwrite_call_result_74300 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), assert_no_overwrite_74293, *[cholesky_74294, list_74295], **kwargs_74299)
        
        
        # ################# End of 'test_cholesky(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cholesky' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_74301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74301)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cholesky'
        return stypy_return_type_74301


    @norecursion
    def test_cho_factor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cho_factor'
        module_type_store = module_type_store.open_function_context('test_cho_factor', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_cho_factor')
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_cho_factor.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_cho_factor', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cho_factor', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cho_factor(...)' code ##################

        
        # Call to assert_no_overwrite(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'cho_factor' (line 175)
        cho_factor_74303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'cho_factor', False)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_74304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        
        # Obtaining an instance of the builtin type 'tuple' (line 175)
        tuple_74305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 175)
        # Adding element type (line 175)
        int_74306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 42), tuple_74305, int_74306)
        # Adding element type (line 175)
        int_74307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 42), tuple_74305, int_74307)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 40), list_74304, tuple_74305)
        
        # Processing the call keyword arguments (line 175)
        kwargs_74308 = {}
        # Getting the type of 'assert_no_overwrite' (line 175)
        assert_no_overwrite_74302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'assert_no_overwrite', False)
        # Calling assert_no_overwrite(args, kwargs) (line 175)
        assert_no_overwrite_call_result_74309 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), assert_no_overwrite_74302, *[cho_factor_74303, list_74304], **kwargs_74308)
        
        
        # ################# End of 'test_cho_factor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cho_factor' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_74310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cho_factor'
        return stypy_return_type_74310


    @norecursion
    def test_cho_solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cho_solve'
        module_type_store = module_type_store.open_function_context('test_cho_solve', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_cho_solve')
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_cho_solve.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_cho_solve', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cho_solve', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cho_solve(...)' code ##################

        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to array(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_74312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_74313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        int_74314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 19), list_74313, int_74314)
        # Adding element type (line 178)
        int_74315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 19), list_74313, int_74315)
        # Adding element type (line 178)
        int_74316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 19), list_74313, int_74316)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 18), list_74312, list_74313)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_74317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        int_74318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 31), list_74317, int_74318)
        # Adding element type (line 178)
        int_74319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 31), list_74317, int_74319)
        # Adding element type (line 178)
        int_74320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 31), list_74317, int_74320)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 18), list_74312, list_74317)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_74321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        int_74322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 44), list_74321, int_74322)
        # Adding element type (line 178)
        int_74323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 44), list_74321, int_74323)
        # Adding element type (line 178)
        int_74324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 44), list_74321, int_74324)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 18), list_74312, list_74321)
        
        # Processing the call keyword arguments (line 178)
        kwargs_74325 = {}
        # Getting the type of 'array' (line 178)
        array_74311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'array', False)
        # Calling array(args, kwargs) (line 178)
        array_call_result_74326 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), array_74311, *[list_74312], **kwargs_74325)
        
        # Assigning a type to the variable 'x' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'x', array_call_result_74326)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to cho_factor(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'x' (line 179)
        x_74328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 26), 'x', False)
        # Processing the call keyword arguments (line 179)
        kwargs_74329 = {}
        # Getting the type of 'cho_factor' (line 179)
        cho_factor_74327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'cho_factor', False)
        # Calling cho_factor(args, kwargs) (line 179)
        cho_factor_call_result_74330 = invoke(stypy.reporting.localization.Localization(__file__, 179, 15), cho_factor_74327, *[x_74328], **kwargs_74329)
        
        # Assigning a type to the variable 'xcho' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'xcho', cho_factor_call_result_74330)
        
        # Call to assert_no_overwrite(...): (line 180)
        # Processing the call arguments (line 180)

        @norecursion
        def _stypy_temp_lambda_28(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_28'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_28', 180, 28, True)
            # Passed parameters checking function
            _stypy_temp_lambda_28.stypy_localization = localization
            _stypy_temp_lambda_28.stypy_type_of_self = None
            _stypy_temp_lambda_28.stypy_type_store = module_type_store
            _stypy_temp_lambda_28.stypy_function_name = '_stypy_temp_lambda_28'
            _stypy_temp_lambda_28.stypy_param_names_list = ['b']
            _stypy_temp_lambda_28.stypy_varargs_param_name = None
            _stypy_temp_lambda_28.stypy_kwargs_param_name = None
            _stypy_temp_lambda_28.stypy_call_defaults = defaults
            _stypy_temp_lambda_28.stypy_call_varargs = varargs
            _stypy_temp_lambda_28.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_28', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_28', ['b'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to cho_solve(...): (line 180)
            # Processing the call arguments (line 180)
            # Getting the type of 'xcho' (line 180)
            xcho_74333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 48), 'xcho', False)
            # Getting the type of 'b' (line 180)
            b_74334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 54), 'b', False)
            # Processing the call keyword arguments (line 180)
            kwargs_74335 = {}
            # Getting the type of 'cho_solve' (line 180)
            cho_solve_74332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 38), 'cho_solve', False)
            # Calling cho_solve(args, kwargs) (line 180)
            cho_solve_call_result_74336 = invoke(stypy.reporting.localization.Localization(__file__, 180, 38), cho_solve_74332, *[xcho_74333, b_74334], **kwargs_74335)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'stypy_return_type', cho_solve_call_result_74336)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_28' in the type store
            # Getting the type of 'stypy_return_type' (line 180)
            stypy_return_type_74337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_74337)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_28'
            return stypy_return_type_74337

        # Assigning a type to the variable '_stypy_temp_lambda_28' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), '_stypy_temp_lambda_28', _stypy_temp_lambda_28)
        # Getting the type of '_stypy_temp_lambda_28' (line 180)
        _stypy_temp_lambda_28_74338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), '_stypy_temp_lambda_28')
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_74339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        # Adding element type (line 180)
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_74340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        int_74341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 60), tuple_74340, int_74341)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 58), list_74339, tuple_74340)
        
        # Processing the call keyword arguments (line 180)
        kwargs_74342 = {}
        # Getting the type of 'assert_no_overwrite' (line 180)
        assert_no_overwrite_74331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'assert_no_overwrite', False)
        # Calling assert_no_overwrite(args, kwargs) (line 180)
        assert_no_overwrite_call_result_74343 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), assert_no_overwrite_74331, *[_stypy_temp_lambda_28_74338, list_74339], **kwargs_74342)
        
        
        # ################# End of 'test_cho_solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cho_solve' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_74344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74344)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cho_solve'
        return stypy_return_type_74344


    @norecursion
    def test_cholesky_banded(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cholesky_banded'
        module_type_store = module_type_store.open_function_context('test_cholesky_banded', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_cholesky_banded')
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_cholesky_banded.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_cholesky_banded', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cholesky_banded', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cholesky_banded(...)' code ##################

        
        # Call to assert_no_overwrite(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'cholesky_banded' (line 183)
        cholesky_banded_74346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'cholesky_banded', False)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_74347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 183)
        tuple_74348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 183)
        # Adding element type (line 183)
        int_74349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 47), tuple_74348, int_74349)
        # Adding element type (line 183)
        int_74350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 47), tuple_74348, int_74350)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 45), list_74347, tuple_74348)
        
        # Processing the call keyword arguments (line 183)
        kwargs_74351 = {}
        # Getting the type of 'assert_no_overwrite' (line 183)
        assert_no_overwrite_74345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'assert_no_overwrite', False)
        # Calling assert_no_overwrite(args, kwargs) (line 183)
        assert_no_overwrite_call_result_74352 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), assert_no_overwrite_74345, *[cholesky_banded_74346, list_74347], **kwargs_74351)
        
        
        # ################# End of 'test_cholesky_banded(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cholesky_banded' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_74353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cholesky_banded'
        return stypy_return_type_74353


    @norecursion
    def test_cho_solve_banded(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cho_solve_banded'
        module_type_store = module_type_store.open_function_context('test_cho_solve_banded', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_cho_solve_banded')
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_cho_solve_banded.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_cho_solve_banded', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cho_solve_banded', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cho_solve_banded(...)' code ##################

        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to array(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_74355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        # Adding element type (line 186)
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_74356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        # Adding element type (line 186)
        int_74357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 19), list_74356, int_74357)
        # Adding element type (line 186)
        int_74358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 19), list_74356, int_74358)
        # Adding element type (line 186)
        int_74359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 19), list_74356, int_74359)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 18), list_74355, list_74356)
        # Adding element type (line 186)
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_74360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        # Adding element type (line 186)
        int_74361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 32), list_74360, int_74361)
        # Adding element type (line 186)
        int_74362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 32), list_74360, int_74362)
        # Adding element type (line 186)
        int_74363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 32), list_74360, int_74363)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 18), list_74355, list_74360)
        
        # Processing the call keyword arguments (line 186)
        kwargs_74364 = {}
        # Getting the type of 'array' (line 186)
        array_74354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'array', False)
        # Calling array(args, kwargs) (line 186)
        array_call_result_74365 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), array_74354, *[list_74355], **kwargs_74364)
        
        # Assigning a type to the variable 'x' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'x', array_call_result_74365)
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to cholesky_banded(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'x' (line 187)
        x_74367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'x', False)
        # Processing the call keyword arguments (line 187)
        kwargs_74368 = {}
        # Getting the type of 'cholesky_banded' (line 187)
        cholesky_banded_74366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'cholesky_banded', False)
        # Calling cholesky_banded(args, kwargs) (line 187)
        cholesky_banded_call_result_74369 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), cholesky_banded_74366, *[x_74367], **kwargs_74368)
        
        # Assigning a type to the variable 'xcho' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'xcho', cholesky_banded_call_result_74369)
        
        # Call to assert_no_overwrite(...): (line 188)
        # Processing the call arguments (line 188)

        @norecursion
        def _stypy_temp_lambda_29(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_29'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_29', 188, 28, True)
            # Passed parameters checking function
            _stypy_temp_lambda_29.stypy_localization = localization
            _stypy_temp_lambda_29.stypy_type_of_self = None
            _stypy_temp_lambda_29.stypy_type_store = module_type_store
            _stypy_temp_lambda_29.stypy_function_name = '_stypy_temp_lambda_29'
            _stypy_temp_lambda_29.stypy_param_names_list = ['b']
            _stypy_temp_lambda_29.stypy_varargs_param_name = None
            _stypy_temp_lambda_29.stypy_kwargs_param_name = None
            _stypy_temp_lambda_29.stypy_call_defaults = defaults
            _stypy_temp_lambda_29.stypy_call_varargs = varargs
            _stypy_temp_lambda_29.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_29', ['b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_29', ['b'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to cho_solve_banded(...): (line 188)
            # Processing the call arguments (line 188)
            
            # Obtaining an instance of the builtin type 'tuple' (line 188)
            tuple_74372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 188)
            # Adding element type (line 188)
            # Getting the type of 'xcho' (line 188)
            xcho_74373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 56), 'xcho', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 56), tuple_74372, xcho_74373)
            # Adding element type (line 188)
            # Getting the type of 'False' (line 188)
            False_74374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 62), 'False', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 56), tuple_74372, False_74374)
            
            # Getting the type of 'b' (line 188)
            b_74375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 70), 'b', False)
            # Processing the call keyword arguments (line 188)
            kwargs_74376 = {}
            # Getting the type of 'cho_solve_banded' (line 188)
            cho_solve_banded_74371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 38), 'cho_solve_banded', False)
            # Calling cho_solve_banded(args, kwargs) (line 188)
            cho_solve_banded_call_result_74377 = invoke(stypy.reporting.localization.Localization(__file__, 188, 38), cho_solve_banded_74371, *[tuple_74372, b_74375], **kwargs_74376)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'stypy_return_type', cho_solve_banded_call_result_74377)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_29' in the type store
            # Getting the type of 'stypy_return_type' (line 188)
            stypy_return_type_74378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_74378)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_29'
            return stypy_return_type_74378

        # Assigning a type to the variable '_stypy_temp_lambda_29' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), '_stypy_temp_lambda_29', _stypy_temp_lambda_29)
        # Getting the type of '_stypy_temp_lambda_29' (line 188)
        _stypy_temp_lambda_29_74379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), '_stypy_temp_lambda_29')
        
        # Obtaining an instance of the builtin type 'list' (line 189)
        list_74380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 189)
        # Adding element type (line 189)
        
        # Obtaining an instance of the builtin type 'tuple' (line 189)
        tuple_74381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 189)
        # Adding element type (line 189)
        int_74382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 30), tuple_74381, int_74382)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 28), list_74380, tuple_74381)
        
        # Processing the call keyword arguments (line 188)
        kwargs_74383 = {}
        # Getting the type of 'assert_no_overwrite' (line 188)
        assert_no_overwrite_74370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'assert_no_overwrite', False)
        # Calling assert_no_overwrite(args, kwargs) (line 188)
        assert_no_overwrite_call_result_74384 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), assert_no_overwrite_74370, *[_stypy_temp_lambda_29_74379, list_74380], **kwargs_74383)
        
        
        # ################# End of 'test_cho_solve_banded(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cho_solve_banded' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_74385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cho_solve_banded'
        return stypy_return_type_74385


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 170, 0, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestOverwrite' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'TestOverwrite', TestOverwrite)
# Declaration of the 'TestEmptyArray' class

class TestEmptyArray(object, ):

    @norecursion
    def test_cho_factor_empty_square(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cho_factor_empty_square'
        module_type_store = module_type_store.open_function_context('test_cho_factor_empty_square', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_localization', localization)
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_function_name', 'TestEmptyArray.test_cho_factor_empty_square')
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_param_names_list', [])
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestEmptyArray.test_cho_factor_empty_square.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEmptyArray.test_cho_factor_empty_square', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cho_factor_empty_square', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cho_factor_empty_square(...)' code ##################

        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to empty(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Obtaining an instance of the builtin type 'tuple' (line 194)
        tuple_74387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 194)
        # Adding element type (line 194)
        int_74388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 19), tuple_74387, int_74388)
        # Adding element type (line 194)
        int_74389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 19), tuple_74387, int_74389)
        
        # Processing the call keyword arguments (line 194)
        kwargs_74390 = {}
        # Getting the type of 'empty' (line 194)
        empty_74386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'empty', False)
        # Calling empty(args, kwargs) (line 194)
        empty_call_result_74391 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), empty_74386, *[tuple_74387], **kwargs_74390)
        
        # Assigning a type to the variable 'a' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'a', empty_call_result_74391)
        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to array(...): (line 195)
        # Processing the call arguments (line 195)
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_74393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        
        # Processing the call keyword arguments (line 195)
        kwargs_74394 = {}
        # Getting the type of 'array' (line 195)
        array_74392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'array', False)
        # Calling array(args, kwargs) (line 195)
        array_call_result_74395 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), array_74392, *[list_74393], **kwargs_74394)
        
        # Assigning a type to the variable 'b' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'b', array_call_result_74395)
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to array(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_74397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_74398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 18), list_74397, list_74398)
        
        # Processing the call keyword arguments (line 196)
        kwargs_74399 = {}
        # Getting the type of 'array' (line 196)
        array_74396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'array', False)
        # Calling array(args, kwargs) (line 196)
        array_call_result_74400 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), array_74396, *[list_74397], **kwargs_74399)
        
        # Assigning a type to the variable 'c' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'c', array_call_result_74400)
        
        # Assigning a List to a Name (line 197):
        
        # Assigning a List to a Name (line 197):
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_74401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        
        # Assigning a type to the variable 'd' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'd', list_74401)
        
        # Assigning a List to a Name (line 198):
        
        # Assigning a List to a Name (line 198):
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_74402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        # Adding element type (line 198)
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_74403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 12), list_74402, list_74403)
        
        # Assigning a type to the variable 'e' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'e', list_74402)
        
        # Assigning a Call to a Tuple (line 200):
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_74404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        
        # Call to cho_factor(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'a' (line 200)
        a_74406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'a', False)
        # Processing the call keyword arguments (line 200)
        kwargs_74407 = {}
        # Getting the type of 'cho_factor' (line 200)
        cho_factor_74405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'cho_factor', False)
        # Calling cho_factor(args, kwargs) (line 200)
        cho_factor_call_result_74408 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), cho_factor_74405, *[a_74406], **kwargs_74407)
        
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___74409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), cho_factor_call_result_74408, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_74410 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___74409, int_74404)
        
        # Assigning a type to the variable 'tuple_var_assignment_73299' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_73299', subscript_call_result_74410)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_74411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        
        # Call to cho_factor(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'a' (line 200)
        a_74413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'a', False)
        # Processing the call keyword arguments (line 200)
        kwargs_74414 = {}
        # Getting the type of 'cho_factor' (line 200)
        cho_factor_74412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'cho_factor', False)
        # Calling cho_factor(args, kwargs) (line 200)
        cho_factor_call_result_74415 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), cho_factor_74412, *[a_74413], **kwargs_74414)
        
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___74416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), cho_factor_call_result_74415, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_74417 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___74416, int_74411)
        
        # Assigning a type to the variable 'tuple_var_assignment_73300' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_73300', subscript_call_result_74417)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_73299' (line 200)
        tuple_var_assignment_73299_74418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_73299')
        # Assigning a type to the variable 'x' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'x', tuple_var_assignment_73299_74418)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_73300' (line 200)
        tuple_var_assignment_73300_74419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_73300')
        # Assigning a type to the variable '_' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), '_', tuple_var_assignment_73300_74419)
        
        # Call to assert_array_equal(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'x' (line 201)
        x_74421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'x', False)
        # Getting the type of 'a' (line 201)
        a_74422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 30), 'a', False)
        # Processing the call keyword arguments (line 201)
        kwargs_74423 = {}
        # Getting the type of 'assert_array_equal' (line 201)
        assert_array_equal_74420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 201)
        assert_array_equal_call_result_74424 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), assert_array_equal_74420, *[x_74421, a_74422], **kwargs_74423)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_74425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        # Getting the type of 'b' (line 203)
        b_74426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 18), list_74425, b_74426)
        # Adding element type (line 203)
        # Getting the type of 'c' (line 203)
        c_74427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 18), list_74425, c_74427)
        # Adding element type (line 203)
        # Getting the type of 'd' (line 203)
        d_74428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 18), list_74425, d_74428)
        # Adding element type (line 203)
        # Getting the type of 'e' (line 203)
        e_74429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 28), 'e')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 18), list_74425, e_74429)
        
        # Testing the type of a for loop iterable (line 203)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 203, 8), list_74425)
        # Getting the type of the for loop variable (line 203)
        for_loop_var_74430 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 203, 8), list_74425)
        # Assigning a type to the variable 'x' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'x', for_loop_var_74430)
        # SSA begins for a for statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_raises(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'ValueError' (line 204)
        ValueError_74432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'ValueError', False)
        # Getting the type of 'cho_factor' (line 204)
        cho_factor_74433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 38), 'cho_factor', False)
        # Getting the type of 'x' (line 204)
        x_74434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 50), 'x', False)
        # Processing the call keyword arguments (line 204)
        kwargs_74435 = {}
        # Getting the type of 'assert_raises' (line 204)
        assert_raises_74431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 204)
        assert_raises_call_result_74436 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), assert_raises_74431, *[ValueError_74432, cho_factor_74433, x_74434], **kwargs_74435)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cho_factor_empty_square(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cho_factor_empty_square' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_74437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74437)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cho_factor_empty_square'
        return stypy_return_type_74437


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 192, 0, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEmptyArray.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestEmptyArray' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'TestEmptyArray', TestEmptyArray)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
