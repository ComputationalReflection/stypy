
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from numpy.testing import assert_, assert_allclose, assert_equal
4: from pytest import raises as assert_raises
5: import numpy as np
6: 
7: from scipy.optimize._lsq.common import (
8:     step_size_to_bound, find_active_constraints, make_strictly_feasible,
9:     CL_scaling_vector, intersect_trust_region, build_quadratic_1d,
10:     minimize_quadratic_1d, evaluate_quadratic, reflective_transformation)
11: 
12: 
13: class TestBounds(object):
14:     def test_step_size_to_bounds(self):
15:         lb = np.array([-1.0, 2.5, 10.0])
16:         ub = np.array([1.0, 5.0, 100.0])
17:         x = np.array([0.0, 2.5, 12.0])
18: 
19:         s = np.array([0.1, 0.0, 0.0])
20:         step, hits = step_size_to_bound(x, s, lb, ub)
21:         assert_equal(step, 10)
22:         assert_equal(hits, [1, 0, 0])
23: 
24:         s = np.array([0.01, 0.05, -1.0])
25:         step, hits = step_size_to_bound(x, s, lb, ub)
26:         assert_equal(step, 2)
27:         assert_equal(hits, [0, 0, -1])
28: 
29:         s = np.array([10.0, -0.0001, 100.0])
30:         step, hits = step_size_to_bound(x, s, lb, ub)
31:         assert_equal(step, np.array(-0))
32:         assert_equal(hits, [0, -1, 0])
33: 
34:         s = np.array([1.0, 0.5, -2.0])
35:         step, hits = step_size_to_bound(x, s, lb, ub)
36:         assert_equal(step, 1.0)
37:         assert_equal(hits, [1, 0, -1])
38: 
39:         s = np.zeros(3)
40:         step, hits = step_size_to_bound(x, s, lb, ub)
41:         assert_equal(step, np.inf)
42:         assert_equal(hits, [0, 0, 0])
43: 
44:     def test_find_active_constraints(self):
45:         lb = np.array([0.0, -10.0, 1.0])
46:         ub = np.array([1.0, 0.0, 100.0])
47: 
48:         x = np.array([0.5, -5.0, 2.0])
49:         active = find_active_constraints(x, lb, ub)
50:         assert_equal(active, [0, 0, 0])
51: 
52:         x = np.array([0.0, 0.0, 10.0])
53:         active = find_active_constraints(x, lb, ub)
54:         assert_equal(active, [-1, 1, 0])
55: 
56:         active = find_active_constraints(x, lb, ub, rtol=0)
57:         assert_equal(active, [-1, 1, 0])
58: 
59:         x = np.array([1e-9, -1e-8, 100 - 1e-9])
60:         active = find_active_constraints(x, lb, ub)
61:         assert_equal(active, [0, 0, 1])
62: 
63:         active = find_active_constraints(x, lb, ub, rtol=1.5e-9)
64:         assert_equal(active, [-1, 0, 1])
65: 
66:         lb = np.array([1.0, -np.inf, -np.inf])
67:         ub = np.array([np.inf, 10.0, np.inf])
68: 
69:         x = np.ones(3)
70:         active = find_active_constraints(x, lb, ub)
71:         assert_equal(active, [-1, 0, 0])
72: 
73:         # Handles out-of-bound cases.
74:         x = np.array([0.0, 11.0, 0.0])
75:         active = find_active_constraints(x, lb, ub)
76:         assert_equal(active, [-1, 1, 0])
77: 
78:         active = find_active_constraints(x, lb, ub, rtol=0)
79:         assert_equal(active, [-1, 1, 0])
80: 
81:     def test_make_strictly_feasible(self):
82:         lb = np.array([-0.5, -0.8, 2.0])
83:         ub = np.array([0.8, 1.0, 3.0])
84: 
85:         x = np.array([-0.5, 0.0, 2 + 1e-10])
86: 
87:         x_new = make_strictly_feasible(x, lb, ub, rstep=0)
88:         assert_(x_new[0] > -0.5)
89:         assert_equal(x_new[1:], x[1:])
90: 
91:         x_new = make_strictly_feasible(x, lb, ub, rstep=1e-4)
92:         assert_equal(x_new, [-0.5 + 1e-4, 0.0, 2 * (1 + 1e-4)])
93: 
94:         x = np.array([-0.5, -1, 3.1])
95:         x_new = make_strictly_feasible(x, lb, ub)
96:         assert_(np.all((x_new >= lb) & (x_new <= ub)))
97: 
98:         x_new = make_strictly_feasible(x, lb, ub, rstep=0)
99:         assert_(np.all((x_new >= lb) & (x_new <= ub)))
100: 
101:         lb = np.array([-1, 100.0])
102:         ub = np.array([1, 100.0 + 1e-10])
103:         x = np.array([0, 100.0])
104:         x_new = make_strictly_feasible(x, lb, ub, rstep=1e-8)
105:         assert_equal(x_new, [0, 100.0 + 0.5e-10])
106: 
107:     def test_scaling_vector(self):
108:         lb = np.array([-np.inf, -5.0, 1.0, -np.inf])
109:         ub = np.array([1.0, np.inf, 10.0, np.inf])
110:         x = np.array([0.5, 2.0, 5.0, 0.0])
111:         g = np.array([1.0, 0.1, -10.0, 0.0])
112:         v, dv = CL_scaling_vector(x, g, lb, ub)
113:         assert_equal(v, [1.0, 7.0, 5.0, 1.0])
114:         assert_equal(dv, [0.0, 1.0, -1.0, 0.0])
115: 
116: 
117: class TestQuadraticFunction(object):
118:     def setup_method(self):
119:         self.J = np.array([
120:             [0.1, 0.2],
121:             [-1.0, 1.0],
122:             [0.5, 0.2]])
123:         self.g = np.array([0.8, -2.0])
124:         self.diag = np.array([1.0, 2.0])
125: 
126:     def test_build_quadratic_1d(self):
127:         s = np.zeros(2)
128:         a, b = build_quadratic_1d(self.J, self.g, s)
129:         assert_equal(a, 0)
130:         assert_equal(b, 0)
131: 
132:         a, b = build_quadratic_1d(self.J, self.g, s, diag=self.diag)
133:         assert_equal(a, 0)
134:         assert_equal(b, 0)
135: 
136:         s = np.array([1.0, -1.0])
137:         a, b = build_quadratic_1d(self.J, self.g, s)
138:         assert_equal(a, 2.05)
139:         assert_equal(b, 2.8)
140: 
141:         a, b = build_quadratic_1d(self.J, self.g, s, diag=self.diag)
142:         assert_equal(a, 3.55)
143:         assert_equal(b, 2.8)
144: 
145:         s0 = np.array([0.5, 0.5])
146:         a, b, c = build_quadratic_1d(self.J, self.g, s, diag=self.diag, s0=s0)
147:         assert_equal(a, 3.55)
148:         assert_allclose(b, 2.39)
149:         assert_allclose(c, -0.1525)
150: 
151:     def test_minimize_quadratic_1d(self):
152:         a = 5
153:         b = -1
154: 
155:         t, y = minimize_quadratic_1d(a, b, 1, 2)
156:         assert_equal(t, 1)
157:         assert_equal(y, a * t**2 + b * t)
158: 
159:         t, y = minimize_quadratic_1d(a, b, -2, -1)
160:         assert_equal(t, -1)
161:         assert_equal(y, a * t**2 + b * t)
162: 
163:         t, y = minimize_quadratic_1d(a, b, -1, 1)
164:         assert_equal(t, 0.1)
165:         assert_equal(y, a * t**2 + b * t)
166: 
167:         c = 10
168:         t, y = minimize_quadratic_1d(a, b, -1, 1, c=c)
169:         assert_equal(t, 0.1)
170:         assert_equal(y, a * t**2 + b * t + c)
171: 
172:     def test_evaluate_quadratic(self):
173:         s = np.array([1.0, -1.0])
174: 
175:         value = evaluate_quadratic(self.J, self.g, s)
176:         assert_equal(value, 4.85)
177: 
178:         value = evaluate_quadratic(self.J, self.g, s, diag=self.diag)
179:         assert_equal(value, 6.35)
180: 
181:         s = np.array([[1.0, -1.0],
182:                      [1.0, 1.0],
183:                      [0.0, 0.0]])
184: 
185:         values = evaluate_quadratic(self.J, self.g, s)
186:         assert_allclose(values, [4.85, -0.91, 0.0])
187: 
188:         values = evaluate_quadratic(self.J, self.g, s, diag=self.diag)
189:         assert_allclose(values, [6.35, 0.59, 0.0])
190: 
191: 
192: class TestTrustRegion(object):
193:     def test_intersect(self):
194:         Delta = 1.0
195: 
196:         x = np.zeros(3)
197:         s = np.array([1.0, 0.0, 0.0])
198:         t_neg, t_pos = intersect_trust_region(x, s, Delta)
199:         assert_equal(t_neg, -1)
200:         assert_equal(t_pos, 1)
201: 
202:         s = np.array([-1.0, 1.0, -1.0])
203:         t_neg, t_pos = intersect_trust_region(x, s, Delta)
204:         assert_allclose(t_neg, -3**-0.5)
205:         assert_allclose(t_pos, 3**-0.5)
206: 
207:         x = np.array([0.5, -0.5, 0])
208:         s = np.array([0, 0, 1.0])
209:         t_neg, t_pos = intersect_trust_region(x, s, Delta)
210:         assert_allclose(t_neg, -2**-0.5)
211:         assert_allclose(t_pos, 2**-0.5)
212: 
213:         x = np.ones(3)
214:         assert_raises(ValueError, intersect_trust_region, x, s, Delta)
215: 
216:         x = np.zeros(3)
217:         s = np.zeros(3)
218:         assert_raises(ValueError, intersect_trust_region, x, s, Delta)
219: 
220: 
221: def test_reflective_transformation():
222:     lb = np.array([-1, -2], dtype=float)
223:     ub = np.array([5, 3], dtype=float)
224: 
225:     y = np.array([0, 0])
226:     x, g = reflective_transformation(y, lb, ub)
227:     assert_equal(x, y)
228:     assert_equal(g, np.ones(2))
229: 
230:     y = np.array([-4, 4], dtype=float)
231: 
232:     x, g = reflective_transformation(y, lb, np.array([np.inf, np.inf]))
233:     assert_equal(x, [2, 4])
234:     assert_equal(g, [-1, 1])
235: 
236:     x, g = reflective_transformation(y, np.array([-np.inf, -np.inf]), ub)
237:     assert_equal(x, [-4, 2])
238:     assert_equal(g, [1, -1])
239: 
240:     x, g = reflective_transformation(y, lb, ub)
241:     assert_equal(x, [2, 2])
242:     assert_equal(g, [-1, -1])
243: 
244:     lb = np.array([-np.inf, -2])
245:     ub = np.array([5, np.inf])
246:     y = np.array([10, 10], dtype=float)
247:     x, g = reflective_transformation(y, lb, ub)
248:     assert_equal(x, [0, 10])
249:     assert_equal(g, [-1, 1])
250: 
251: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_, assert_allclose, assert_equal' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_215669 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_215669) is not StypyTypeError):

    if (import_215669 != 'pyd_module'):
        __import__(import_215669)
        sys_modules_215670 = sys.modules[import_215669]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_215670.module_type_store, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_215670, sys_modules_215670.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'], [assert_, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_215669)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from pytest import assert_raises' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_215671 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_215671) is not StypyTypeError):

    if (import_215671 != 'pyd_module'):
        __import__(import_215671)
        sys_modules_215672 = sys.modules[import_215671]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_215672.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_215672, sys_modules_215672.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_215671)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_215673 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_215673) is not StypyTypeError):

    if (import_215673 != 'pyd_module'):
        __import__(import_215673)
        sys_modules_215674 = sys.modules[import_215673]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_215674.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_215673)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.optimize._lsq.common import step_size_to_bound, find_active_constraints, make_strictly_feasible, CL_scaling_vector, intersect_trust_region, build_quadratic_1d, minimize_quadratic_1d, evaluate_quadratic, reflective_transformation' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_215675 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize._lsq.common')

if (type(import_215675) is not StypyTypeError):

    if (import_215675 != 'pyd_module'):
        __import__(import_215675)
        sys_modules_215676 = sys.modules[import_215675]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize._lsq.common', sys_modules_215676.module_type_store, module_type_store, ['step_size_to_bound', 'find_active_constraints', 'make_strictly_feasible', 'CL_scaling_vector', 'intersect_trust_region', 'build_quadratic_1d', 'minimize_quadratic_1d', 'evaluate_quadratic', 'reflective_transformation'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_215676, sys_modules_215676.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import step_size_to_bound, find_active_constraints, make_strictly_feasible, CL_scaling_vector, intersect_trust_region, build_quadratic_1d, minimize_quadratic_1d, evaluate_quadratic, reflective_transformation

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['step_size_to_bound', 'find_active_constraints', 'make_strictly_feasible', 'CL_scaling_vector', 'intersect_trust_region', 'build_quadratic_1d', 'minimize_quadratic_1d', 'evaluate_quadratic', 'reflective_transformation'], [step_size_to_bound, find_active_constraints, make_strictly_feasible, CL_scaling_vector, intersect_trust_region, build_quadratic_1d, minimize_quadratic_1d, evaluate_quadratic, reflective_transformation])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize._lsq.common', import_215675)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'TestBounds' class

class TestBounds(object, ):

    @norecursion
    def test_step_size_to_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_step_size_to_bounds'
        module_type_store = module_type_store.open_function_context('test_step_size_to_bounds', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_localization', localization)
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_function_name', 'TestBounds.test_step_size_to_bounds')
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBounds.test_step_size_to_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBounds.test_step_size_to_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_step_size_to_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_step_size_to_bounds(...)' code ##################

        
        # Assigning a Call to a Name (line 15):
        
        # Assigning a Call to a Name (line 15):
        
        # Call to array(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_215679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        # Adding element type (line 15)
        float_215680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_215679, float_215680)
        # Adding element type (line 15)
        float_215681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_215679, float_215681)
        # Adding element type (line 15)
        float_215682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_215679, float_215682)
        
        # Processing the call keyword arguments (line 15)
        kwargs_215683 = {}
        # Getting the type of 'np' (line 15)
        np_215677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 15)
        array_215678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 13), np_215677, 'array')
        # Calling array(args, kwargs) (line 15)
        array_call_result_215684 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), array_215678, *[list_215679], **kwargs_215683)
        
        # Assigning a type to the variable 'lb' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'lb', array_call_result_215684)
        
        # Assigning a Call to a Name (line 16):
        
        # Assigning a Call to a Name (line 16):
        
        # Call to array(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_215687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        # Adding element type (line 16)
        float_215688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_215687, float_215688)
        # Adding element type (line 16)
        float_215689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_215687, float_215689)
        # Adding element type (line 16)
        float_215690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), list_215687, float_215690)
        
        # Processing the call keyword arguments (line 16)
        kwargs_215691 = {}
        # Getting the type of 'np' (line 16)
        np_215685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 16)
        array_215686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 13), np_215685, 'array')
        # Calling array(args, kwargs) (line 16)
        array_call_result_215692 = invoke(stypy.reporting.localization.Localization(__file__, 16, 13), array_215686, *[list_215687], **kwargs_215691)
        
        # Assigning a type to the variable 'ub' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'ub', array_call_result_215692)
        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Call to array(...): (line 17)
        # Processing the call arguments (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_215695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        float_215696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 21), list_215695, float_215696)
        # Adding element type (line 17)
        float_215697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 21), list_215695, float_215697)
        # Adding element type (line 17)
        float_215698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 21), list_215695, float_215698)
        
        # Processing the call keyword arguments (line 17)
        kwargs_215699 = {}
        # Getting the type of 'np' (line 17)
        np_215693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 17)
        array_215694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), np_215693, 'array')
        # Calling array(args, kwargs) (line 17)
        array_call_result_215700 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), array_215694, *[list_215695], **kwargs_215699)
        
        # Assigning a type to the variable 'x' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'x', array_call_result_215700)
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to array(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_215703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        float_215704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), list_215703, float_215704)
        # Adding element type (line 19)
        float_215705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), list_215703, float_215705)
        # Adding element type (line 19)
        float_215706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), list_215703, float_215706)
        
        # Processing the call keyword arguments (line 19)
        kwargs_215707 = {}
        # Getting the type of 'np' (line 19)
        np_215701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 19)
        array_215702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), np_215701, 'array')
        # Calling array(args, kwargs) (line 19)
        array_call_result_215708 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), array_215702, *[list_215703], **kwargs_215707)
        
        # Assigning a type to the variable 's' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 's', array_call_result_215708)
        
        # Assigning a Call to a Tuple (line 20):
        
        # Assigning a Subscript to a Name (line 20):
        
        # Obtaining the type of the subscript
        int_215709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'x' (line 20)
        x_215711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 40), 'x', False)
        # Getting the type of 's' (line 20)
        s_215712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 43), 's', False)
        # Getting the type of 'lb' (line 20)
        lb_215713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 46), 'lb', False)
        # Getting the type of 'ub' (line 20)
        ub_215714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 50), 'ub', False)
        # Processing the call keyword arguments (line 20)
        kwargs_215715 = {}
        # Getting the type of 'step_size_to_bound' (line 20)
        step_size_to_bound_215710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 20)
        step_size_to_bound_call_result_215716 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), step_size_to_bound_215710, *[x_215711, s_215712, lb_215713, ub_215714], **kwargs_215715)
        
        # Obtaining the member '__getitem__' of a type (line 20)
        getitem___215717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), step_size_to_bound_call_result_215716, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 20)
        subscript_call_result_215718 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), getitem___215717, int_215709)
        
        # Assigning a type to the variable 'tuple_var_assignment_215622' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple_var_assignment_215622', subscript_call_result_215718)
        
        # Assigning a Subscript to a Name (line 20):
        
        # Obtaining the type of the subscript
        int_215719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'x' (line 20)
        x_215721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 40), 'x', False)
        # Getting the type of 's' (line 20)
        s_215722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 43), 's', False)
        # Getting the type of 'lb' (line 20)
        lb_215723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 46), 'lb', False)
        # Getting the type of 'ub' (line 20)
        ub_215724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 50), 'ub', False)
        # Processing the call keyword arguments (line 20)
        kwargs_215725 = {}
        # Getting the type of 'step_size_to_bound' (line 20)
        step_size_to_bound_215720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 20)
        step_size_to_bound_call_result_215726 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), step_size_to_bound_215720, *[x_215721, s_215722, lb_215723, ub_215724], **kwargs_215725)
        
        # Obtaining the member '__getitem__' of a type (line 20)
        getitem___215727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), step_size_to_bound_call_result_215726, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 20)
        subscript_call_result_215728 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), getitem___215727, int_215719)
        
        # Assigning a type to the variable 'tuple_var_assignment_215623' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple_var_assignment_215623', subscript_call_result_215728)
        
        # Assigning a Name to a Name (line 20):
        # Getting the type of 'tuple_var_assignment_215622' (line 20)
        tuple_var_assignment_215622_215729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple_var_assignment_215622')
        # Assigning a type to the variable 'step' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'step', tuple_var_assignment_215622_215729)
        
        # Assigning a Name to a Name (line 20):
        # Getting the type of 'tuple_var_assignment_215623' (line 20)
        tuple_var_assignment_215623_215730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple_var_assignment_215623')
        # Assigning a type to the variable 'hits' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'hits', tuple_var_assignment_215623_215730)
        
        # Call to assert_equal(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'step' (line 21)
        step_215732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'step', False)
        int_215733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'int')
        # Processing the call keyword arguments (line 21)
        kwargs_215734 = {}
        # Getting the type of 'assert_equal' (line 21)
        assert_equal_215731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 21)
        assert_equal_call_result_215735 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert_equal_215731, *[step_215732, int_215733], **kwargs_215734)
        
        
        # Call to assert_equal(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'hits' (line 22)
        hits_215737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'hits', False)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_215738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        int_215739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 27), list_215738, int_215739)
        # Adding element type (line 22)
        int_215740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 27), list_215738, int_215740)
        # Adding element type (line 22)
        int_215741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 27), list_215738, int_215741)
        
        # Processing the call keyword arguments (line 22)
        kwargs_215742 = {}
        # Getting the type of 'assert_equal' (line 22)
        assert_equal_215736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 22)
        assert_equal_call_result_215743 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), assert_equal_215736, *[hits_215737, list_215738], **kwargs_215742)
        
        
        # Assigning a Call to a Name (line 24):
        
        # Assigning a Call to a Name (line 24):
        
        # Call to array(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_215746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        float_215747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), list_215746, float_215747)
        # Adding element type (line 24)
        float_215748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), list_215746, float_215748)
        # Adding element type (line 24)
        float_215749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), list_215746, float_215749)
        
        # Processing the call keyword arguments (line 24)
        kwargs_215750 = {}
        # Getting the type of 'np' (line 24)
        np_215744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 24)
        array_215745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), np_215744, 'array')
        # Calling array(args, kwargs) (line 24)
        array_call_result_215751 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), array_215745, *[list_215746], **kwargs_215750)
        
        # Assigning a type to the variable 's' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 's', array_call_result_215751)
        
        # Assigning a Call to a Tuple (line 25):
        
        # Assigning a Subscript to a Name (line 25):
        
        # Obtaining the type of the subscript
        int_215752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'x' (line 25)
        x_215754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 40), 'x', False)
        # Getting the type of 's' (line 25)
        s_215755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 43), 's', False)
        # Getting the type of 'lb' (line 25)
        lb_215756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 46), 'lb', False)
        # Getting the type of 'ub' (line 25)
        ub_215757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 50), 'ub', False)
        # Processing the call keyword arguments (line 25)
        kwargs_215758 = {}
        # Getting the type of 'step_size_to_bound' (line 25)
        step_size_to_bound_215753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 25)
        step_size_to_bound_call_result_215759 = invoke(stypy.reporting.localization.Localization(__file__, 25, 21), step_size_to_bound_215753, *[x_215754, s_215755, lb_215756, ub_215757], **kwargs_215758)
        
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___215760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), step_size_to_bound_call_result_215759, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_215761 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), getitem___215760, int_215752)
        
        # Assigning a type to the variable 'tuple_var_assignment_215624' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_var_assignment_215624', subscript_call_result_215761)
        
        # Assigning a Subscript to a Name (line 25):
        
        # Obtaining the type of the subscript
        int_215762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'x' (line 25)
        x_215764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 40), 'x', False)
        # Getting the type of 's' (line 25)
        s_215765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 43), 's', False)
        # Getting the type of 'lb' (line 25)
        lb_215766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 46), 'lb', False)
        # Getting the type of 'ub' (line 25)
        ub_215767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 50), 'ub', False)
        # Processing the call keyword arguments (line 25)
        kwargs_215768 = {}
        # Getting the type of 'step_size_to_bound' (line 25)
        step_size_to_bound_215763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 25)
        step_size_to_bound_call_result_215769 = invoke(stypy.reporting.localization.Localization(__file__, 25, 21), step_size_to_bound_215763, *[x_215764, s_215765, lb_215766, ub_215767], **kwargs_215768)
        
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___215770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), step_size_to_bound_call_result_215769, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_215771 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), getitem___215770, int_215762)
        
        # Assigning a type to the variable 'tuple_var_assignment_215625' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_var_assignment_215625', subscript_call_result_215771)
        
        # Assigning a Name to a Name (line 25):
        # Getting the type of 'tuple_var_assignment_215624' (line 25)
        tuple_var_assignment_215624_215772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_var_assignment_215624')
        # Assigning a type to the variable 'step' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'step', tuple_var_assignment_215624_215772)
        
        # Assigning a Name to a Name (line 25):
        # Getting the type of 'tuple_var_assignment_215625' (line 25)
        tuple_var_assignment_215625_215773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_var_assignment_215625')
        # Assigning a type to the variable 'hits' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'hits', tuple_var_assignment_215625_215773)
        
        # Call to assert_equal(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'step' (line 26)
        step_215775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'step', False)
        int_215776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'int')
        # Processing the call keyword arguments (line 26)
        kwargs_215777 = {}
        # Getting the type of 'assert_equal' (line 26)
        assert_equal_215774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 26)
        assert_equal_call_result_215778 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assert_equal_215774, *[step_215775, int_215776], **kwargs_215777)
        
        
        # Call to assert_equal(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'hits' (line 27)
        hits_215780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'hits', False)
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_215781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        int_215782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 27), list_215781, int_215782)
        # Adding element type (line 27)
        int_215783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 27), list_215781, int_215783)
        # Adding element type (line 27)
        int_215784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 27), list_215781, int_215784)
        
        # Processing the call keyword arguments (line 27)
        kwargs_215785 = {}
        # Getting the type of 'assert_equal' (line 27)
        assert_equal_215779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 27)
        assert_equal_call_result_215786 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_equal_215779, *[hits_215780, list_215781], **kwargs_215785)
        
        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Call to array(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_215789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        float_215790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_215789, float_215790)
        # Adding element type (line 29)
        float_215791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_215789, float_215791)
        # Adding element type (line 29)
        float_215792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_215789, float_215792)
        
        # Processing the call keyword arguments (line 29)
        kwargs_215793 = {}
        # Getting the type of 'np' (line 29)
        np_215787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 29)
        array_215788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), np_215787, 'array')
        # Calling array(args, kwargs) (line 29)
        array_call_result_215794 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), array_215788, *[list_215789], **kwargs_215793)
        
        # Assigning a type to the variable 's' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 's', array_call_result_215794)
        
        # Assigning a Call to a Tuple (line 30):
        
        # Assigning a Subscript to a Name (line 30):
        
        # Obtaining the type of the subscript
        int_215795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'x' (line 30)
        x_215797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'x', False)
        # Getting the type of 's' (line 30)
        s_215798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 's', False)
        # Getting the type of 'lb' (line 30)
        lb_215799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 46), 'lb', False)
        # Getting the type of 'ub' (line 30)
        ub_215800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 50), 'ub', False)
        # Processing the call keyword arguments (line 30)
        kwargs_215801 = {}
        # Getting the type of 'step_size_to_bound' (line 30)
        step_size_to_bound_215796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 30)
        step_size_to_bound_call_result_215802 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), step_size_to_bound_215796, *[x_215797, s_215798, lb_215799, ub_215800], **kwargs_215801)
        
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___215803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), step_size_to_bound_call_result_215802, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_215804 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), getitem___215803, int_215795)
        
        # Assigning a type to the variable 'tuple_var_assignment_215626' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_215626', subscript_call_result_215804)
        
        # Assigning a Subscript to a Name (line 30):
        
        # Obtaining the type of the subscript
        int_215805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'x' (line 30)
        x_215807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'x', False)
        # Getting the type of 's' (line 30)
        s_215808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 's', False)
        # Getting the type of 'lb' (line 30)
        lb_215809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 46), 'lb', False)
        # Getting the type of 'ub' (line 30)
        ub_215810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 50), 'ub', False)
        # Processing the call keyword arguments (line 30)
        kwargs_215811 = {}
        # Getting the type of 'step_size_to_bound' (line 30)
        step_size_to_bound_215806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 30)
        step_size_to_bound_call_result_215812 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), step_size_to_bound_215806, *[x_215807, s_215808, lb_215809, ub_215810], **kwargs_215811)
        
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___215813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), step_size_to_bound_call_result_215812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_215814 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), getitem___215813, int_215805)
        
        # Assigning a type to the variable 'tuple_var_assignment_215627' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_215627', subscript_call_result_215814)
        
        # Assigning a Name to a Name (line 30):
        # Getting the type of 'tuple_var_assignment_215626' (line 30)
        tuple_var_assignment_215626_215815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_215626')
        # Assigning a type to the variable 'step' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'step', tuple_var_assignment_215626_215815)
        
        # Assigning a Name to a Name (line 30):
        # Getting the type of 'tuple_var_assignment_215627' (line 30)
        tuple_var_assignment_215627_215816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'tuple_var_assignment_215627')
        # Assigning a type to the variable 'hits' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'hits', tuple_var_assignment_215627_215816)
        
        # Call to assert_equal(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'step' (line 31)
        step_215818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'step', False)
        
        # Call to array(...): (line 31)
        # Processing the call arguments (line 31)
        int_215821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_215822 = {}
        # Getting the type of 'np' (line 31)
        np_215819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 31)
        array_215820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 27), np_215819, 'array')
        # Calling array(args, kwargs) (line 31)
        array_call_result_215823 = invoke(stypy.reporting.localization.Localization(__file__, 31, 27), array_215820, *[int_215821], **kwargs_215822)
        
        # Processing the call keyword arguments (line 31)
        kwargs_215824 = {}
        # Getting the type of 'assert_equal' (line 31)
        assert_equal_215817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 31)
        assert_equal_call_result_215825 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert_equal_215817, *[step_215818, array_call_result_215823], **kwargs_215824)
        
        
        # Call to assert_equal(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'hits' (line 32)
        hits_215827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'hits', False)
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_215828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        int_215829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 27), list_215828, int_215829)
        # Adding element type (line 32)
        int_215830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 27), list_215828, int_215830)
        # Adding element type (line 32)
        int_215831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 27), list_215828, int_215831)
        
        # Processing the call keyword arguments (line 32)
        kwargs_215832 = {}
        # Getting the type of 'assert_equal' (line 32)
        assert_equal_215826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 32)
        assert_equal_call_result_215833 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), assert_equal_215826, *[hits_215827, list_215828], **kwargs_215832)
        
        
        # Assigning a Call to a Name (line 34):
        
        # Assigning a Call to a Name (line 34):
        
        # Call to array(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_215836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        float_215837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), list_215836, float_215837)
        # Adding element type (line 34)
        float_215838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), list_215836, float_215838)
        # Adding element type (line 34)
        float_215839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), list_215836, float_215839)
        
        # Processing the call keyword arguments (line 34)
        kwargs_215840 = {}
        # Getting the type of 'np' (line 34)
        np_215834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 34)
        array_215835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), np_215834, 'array')
        # Calling array(args, kwargs) (line 34)
        array_call_result_215841 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), array_215835, *[list_215836], **kwargs_215840)
        
        # Assigning a type to the variable 's' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 's', array_call_result_215841)
        
        # Assigning a Call to a Tuple (line 35):
        
        # Assigning a Subscript to a Name (line 35):
        
        # Obtaining the type of the subscript
        int_215842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'x' (line 35)
        x_215844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'x', False)
        # Getting the type of 's' (line 35)
        s_215845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 's', False)
        # Getting the type of 'lb' (line 35)
        lb_215846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 46), 'lb', False)
        # Getting the type of 'ub' (line 35)
        ub_215847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 50), 'ub', False)
        # Processing the call keyword arguments (line 35)
        kwargs_215848 = {}
        # Getting the type of 'step_size_to_bound' (line 35)
        step_size_to_bound_215843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 35)
        step_size_to_bound_call_result_215849 = invoke(stypy.reporting.localization.Localization(__file__, 35, 21), step_size_to_bound_215843, *[x_215844, s_215845, lb_215846, ub_215847], **kwargs_215848)
        
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___215850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), step_size_to_bound_call_result_215849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_215851 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), getitem___215850, int_215842)
        
        # Assigning a type to the variable 'tuple_var_assignment_215628' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'tuple_var_assignment_215628', subscript_call_result_215851)
        
        # Assigning a Subscript to a Name (line 35):
        
        # Obtaining the type of the subscript
        int_215852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'x' (line 35)
        x_215854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'x', False)
        # Getting the type of 's' (line 35)
        s_215855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 's', False)
        # Getting the type of 'lb' (line 35)
        lb_215856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 46), 'lb', False)
        # Getting the type of 'ub' (line 35)
        ub_215857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 50), 'ub', False)
        # Processing the call keyword arguments (line 35)
        kwargs_215858 = {}
        # Getting the type of 'step_size_to_bound' (line 35)
        step_size_to_bound_215853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 35)
        step_size_to_bound_call_result_215859 = invoke(stypy.reporting.localization.Localization(__file__, 35, 21), step_size_to_bound_215853, *[x_215854, s_215855, lb_215856, ub_215857], **kwargs_215858)
        
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___215860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), step_size_to_bound_call_result_215859, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_215861 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), getitem___215860, int_215852)
        
        # Assigning a type to the variable 'tuple_var_assignment_215629' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'tuple_var_assignment_215629', subscript_call_result_215861)
        
        # Assigning a Name to a Name (line 35):
        # Getting the type of 'tuple_var_assignment_215628' (line 35)
        tuple_var_assignment_215628_215862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'tuple_var_assignment_215628')
        # Assigning a type to the variable 'step' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'step', tuple_var_assignment_215628_215862)
        
        # Assigning a Name to a Name (line 35):
        # Getting the type of 'tuple_var_assignment_215629' (line 35)
        tuple_var_assignment_215629_215863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'tuple_var_assignment_215629')
        # Assigning a type to the variable 'hits' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'hits', tuple_var_assignment_215629_215863)
        
        # Call to assert_equal(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'step' (line 36)
        step_215865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'step', False)
        float_215866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'float')
        # Processing the call keyword arguments (line 36)
        kwargs_215867 = {}
        # Getting the type of 'assert_equal' (line 36)
        assert_equal_215864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 36)
        assert_equal_call_result_215868 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert_equal_215864, *[step_215865, float_215866], **kwargs_215867)
        
        
        # Call to assert_equal(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'hits' (line 37)
        hits_215870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'hits', False)
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_215871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        # Adding element type (line 37)
        int_215872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 27), list_215871, int_215872)
        # Adding element type (line 37)
        int_215873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 27), list_215871, int_215873)
        # Adding element type (line 37)
        int_215874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 27), list_215871, int_215874)
        
        # Processing the call keyword arguments (line 37)
        kwargs_215875 = {}
        # Getting the type of 'assert_equal' (line 37)
        assert_equal_215869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 37)
        assert_equal_call_result_215876 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assert_equal_215869, *[hits_215870, list_215871], **kwargs_215875)
        
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to zeros(...): (line 39)
        # Processing the call arguments (line 39)
        int_215879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'int')
        # Processing the call keyword arguments (line 39)
        kwargs_215880 = {}
        # Getting the type of 'np' (line 39)
        np_215877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 39)
        zeros_215878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), np_215877, 'zeros')
        # Calling zeros(args, kwargs) (line 39)
        zeros_call_result_215881 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), zeros_215878, *[int_215879], **kwargs_215880)
        
        # Assigning a type to the variable 's' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 's', zeros_call_result_215881)
        
        # Assigning a Call to a Tuple (line 40):
        
        # Assigning a Subscript to a Name (line 40):
        
        # Obtaining the type of the subscript
        int_215882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'x' (line 40)
        x_215884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 40), 'x', False)
        # Getting the type of 's' (line 40)
        s_215885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 's', False)
        # Getting the type of 'lb' (line 40)
        lb_215886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 46), 'lb', False)
        # Getting the type of 'ub' (line 40)
        ub_215887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 50), 'ub', False)
        # Processing the call keyword arguments (line 40)
        kwargs_215888 = {}
        # Getting the type of 'step_size_to_bound' (line 40)
        step_size_to_bound_215883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 40)
        step_size_to_bound_call_result_215889 = invoke(stypy.reporting.localization.Localization(__file__, 40, 21), step_size_to_bound_215883, *[x_215884, s_215885, lb_215886, ub_215887], **kwargs_215888)
        
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___215890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), step_size_to_bound_call_result_215889, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_215891 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), getitem___215890, int_215882)
        
        # Assigning a type to the variable 'tuple_var_assignment_215630' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_215630', subscript_call_result_215891)
        
        # Assigning a Subscript to a Name (line 40):
        
        # Obtaining the type of the subscript
        int_215892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'int')
        
        # Call to step_size_to_bound(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'x' (line 40)
        x_215894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 40), 'x', False)
        # Getting the type of 's' (line 40)
        s_215895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 's', False)
        # Getting the type of 'lb' (line 40)
        lb_215896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 46), 'lb', False)
        # Getting the type of 'ub' (line 40)
        ub_215897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 50), 'ub', False)
        # Processing the call keyword arguments (line 40)
        kwargs_215898 = {}
        # Getting the type of 'step_size_to_bound' (line 40)
        step_size_to_bound_215893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'step_size_to_bound', False)
        # Calling step_size_to_bound(args, kwargs) (line 40)
        step_size_to_bound_call_result_215899 = invoke(stypy.reporting.localization.Localization(__file__, 40, 21), step_size_to_bound_215893, *[x_215894, s_215895, lb_215896, ub_215897], **kwargs_215898)
        
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___215900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), step_size_to_bound_call_result_215899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_215901 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), getitem___215900, int_215892)
        
        # Assigning a type to the variable 'tuple_var_assignment_215631' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_215631', subscript_call_result_215901)
        
        # Assigning a Name to a Name (line 40):
        # Getting the type of 'tuple_var_assignment_215630' (line 40)
        tuple_var_assignment_215630_215902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_215630')
        # Assigning a type to the variable 'step' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'step', tuple_var_assignment_215630_215902)
        
        # Assigning a Name to a Name (line 40):
        # Getting the type of 'tuple_var_assignment_215631' (line 40)
        tuple_var_assignment_215631_215903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_215631')
        # Assigning a type to the variable 'hits' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'hits', tuple_var_assignment_215631_215903)
        
        # Call to assert_equal(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'step' (line 41)
        step_215905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'step', False)
        # Getting the type of 'np' (line 41)
        np_215906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'np', False)
        # Obtaining the member 'inf' of a type (line 41)
        inf_215907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 27), np_215906, 'inf')
        # Processing the call keyword arguments (line 41)
        kwargs_215908 = {}
        # Getting the type of 'assert_equal' (line 41)
        assert_equal_215904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 41)
        assert_equal_call_result_215909 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assert_equal_215904, *[step_215905, inf_215907], **kwargs_215908)
        
        
        # Call to assert_equal(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'hits' (line 42)
        hits_215911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'hits', False)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_215912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        int_215913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 27), list_215912, int_215913)
        # Adding element type (line 42)
        int_215914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 27), list_215912, int_215914)
        # Adding element type (line 42)
        int_215915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 27), list_215912, int_215915)
        
        # Processing the call keyword arguments (line 42)
        kwargs_215916 = {}
        # Getting the type of 'assert_equal' (line 42)
        assert_equal_215910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 42)
        assert_equal_call_result_215917 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_equal_215910, *[hits_215911, list_215912], **kwargs_215916)
        
        
        # ################# End of 'test_step_size_to_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_step_size_to_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_215918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_215918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_step_size_to_bounds'
        return stypy_return_type_215918


    @norecursion
    def test_find_active_constraints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_find_active_constraints'
        module_type_store = module_type_store.open_function_context('test_find_active_constraints', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_localization', localization)
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_function_name', 'TestBounds.test_find_active_constraints')
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_param_names_list', [])
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBounds.test_find_active_constraints.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBounds.test_find_active_constraints', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_find_active_constraints', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_find_active_constraints(...)' code ##################

        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to array(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_215921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        # Adding element type (line 45)
        float_215922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), list_215921, float_215922)
        # Adding element type (line 45)
        float_215923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), list_215921, float_215923)
        # Adding element type (line 45)
        float_215924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), list_215921, float_215924)
        
        # Processing the call keyword arguments (line 45)
        kwargs_215925 = {}
        # Getting the type of 'np' (line 45)
        np_215919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 45)
        array_215920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 13), np_215919, 'array')
        # Calling array(args, kwargs) (line 45)
        array_call_result_215926 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), array_215920, *[list_215921], **kwargs_215925)
        
        # Assigning a type to the variable 'lb' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'lb', array_call_result_215926)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to array(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_215929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        float_215930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_215929, float_215930)
        # Adding element type (line 46)
        float_215931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_215929, float_215931)
        # Adding element type (line 46)
        float_215932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_215929, float_215932)
        
        # Processing the call keyword arguments (line 46)
        kwargs_215933 = {}
        # Getting the type of 'np' (line 46)
        np_215927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 46)
        array_215928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), np_215927, 'array')
        # Calling array(args, kwargs) (line 46)
        array_call_result_215934 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), array_215928, *[list_215929], **kwargs_215933)
        
        # Assigning a type to the variable 'ub' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'ub', array_call_result_215934)
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to array(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_215937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        float_215938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 21), list_215937, float_215938)
        # Adding element type (line 48)
        float_215939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 21), list_215937, float_215939)
        # Adding element type (line 48)
        float_215940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 21), list_215937, float_215940)
        
        # Processing the call keyword arguments (line 48)
        kwargs_215941 = {}
        # Getting the type of 'np' (line 48)
        np_215935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 48)
        array_215936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), np_215935, 'array')
        # Calling array(args, kwargs) (line 48)
        array_call_result_215942 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), array_215936, *[list_215937], **kwargs_215941)
        
        # Assigning a type to the variable 'x' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'x', array_call_result_215942)
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to find_active_constraints(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'x' (line 49)
        x_215944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 41), 'x', False)
        # Getting the type of 'lb' (line 49)
        lb_215945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 44), 'lb', False)
        # Getting the type of 'ub' (line 49)
        ub_215946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 48), 'ub', False)
        # Processing the call keyword arguments (line 49)
        kwargs_215947 = {}
        # Getting the type of 'find_active_constraints' (line 49)
        find_active_constraints_215943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'find_active_constraints', False)
        # Calling find_active_constraints(args, kwargs) (line 49)
        find_active_constraints_call_result_215948 = invoke(stypy.reporting.localization.Localization(__file__, 49, 17), find_active_constraints_215943, *[x_215944, lb_215945, ub_215946], **kwargs_215947)
        
        # Assigning a type to the variable 'active' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'active', find_active_constraints_call_result_215948)
        
        # Call to assert_equal(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'active' (line 50)
        active_215950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'active', False)
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_215951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        int_215952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_215951, int_215952)
        # Adding element type (line 50)
        int_215953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_215951, int_215953)
        # Adding element type (line 50)
        int_215954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_215951, int_215954)
        
        # Processing the call keyword arguments (line 50)
        kwargs_215955 = {}
        # Getting the type of 'assert_equal' (line 50)
        assert_equal_215949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 50)
        assert_equal_call_result_215956 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert_equal_215949, *[active_215950, list_215951], **kwargs_215955)
        
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to array(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Obtaining an instance of the builtin type 'list' (line 52)
        list_215959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 52)
        # Adding element type (line 52)
        float_215960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), list_215959, float_215960)
        # Adding element type (line 52)
        float_215961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), list_215959, float_215961)
        # Adding element type (line 52)
        float_215962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), list_215959, float_215962)
        
        # Processing the call keyword arguments (line 52)
        kwargs_215963 = {}
        # Getting the type of 'np' (line 52)
        np_215957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 52)
        array_215958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), np_215957, 'array')
        # Calling array(args, kwargs) (line 52)
        array_call_result_215964 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), array_215958, *[list_215959], **kwargs_215963)
        
        # Assigning a type to the variable 'x' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'x', array_call_result_215964)
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to find_active_constraints(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'x' (line 53)
        x_215966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'x', False)
        # Getting the type of 'lb' (line 53)
        lb_215967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 44), 'lb', False)
        # Getting the type of 'ub' (line 53)
        ub_215968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 48), 'ub', False)
        # Processing the call keyword arguments (line 53)
        kwargs_215969 = {}
        # Getting the type of 'find_active_constraints' (line 53)
        find_active_constraints_215965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'find_active_constraints', False)
        # Calling find_active_constraints(args, kwargs) (line 53)
        find_active_constraints_call_result_215970 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), find_active_constraints_215965, *[x_215966, lb_215967, ub_215968], **kwargs_215969)
        
        # Assigning a type to the variable 'active' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'active', find_active_constraints_call_result_215970)
        
        # Call to assert_equal(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'active' (line 54)
        active_215972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'active', False)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_215973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_215974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 29), list_215973, int_215974)
        # Adding element type (line 54)
        int_215975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 29), list_215973, int_215975)
        # Adding element type (line 54)
        int_215976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 29), list_215973, int_215976)
        
        # Processing the call keyword arguments (line 54)
        kwargs_215977 = {}
        # Getting the type of 'assert_equal' (line 54)
        assert_equal_215971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 54)
        assert_equal_call_result_215978 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert_equal_215971, *[active_215972, list_215973], **kwargs_215977)
        
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to find_active_constraints(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'x' (line 56)
        x_215980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'x', False)
        # Getting the type of 'lb' (line 56)
        lb_215981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), 'lb', False)
        # Getting the type of 'ub' (line 56)
        ub_215982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 48), 'ub', False)
        # Processing the call keyword arguments (line 56)
        int_215983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 57), 'int')
        keyword_215984 = int_215983
        kwargs_215985 = {'rtol': keyword_215984}
        # Getting the type of 'find_active_constraints' (line 56)
        find_active_constraints_215979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'find_active_constraints', False)
        # Calling find_active_constraints(args, kwargs) (line 56)
        find_active_constraints_call_result_215986 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), find_active_constraints_215979, *[x_215980, lb_215981, ub_215982], **kwargs_215985)
        
        # Assigning a type to the variable 'active' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'active', find_active_constraints_call_result_215986)
        
        # Call to assert_equal(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'active' (line 57)
        active_215988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'active', False)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_215989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_215990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), list_215989, int_215990)
        # Adding element type (line 57)
        int_215991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), list_215989, int_215991)
        # Adding element type (line 57)
        int_215992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), list_215989, int_215992)
        
        # Processing the call keyword arguments (line 57)
        kwargs_215993 = {}
        # Getting the type of 'assert_equal' (line 57)
        assert_equal_215987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 57)
        assert_equal_call_result_215994 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assert_equal_215987, *[active_215988, list_215989], **kwargs_215993)
        
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to array(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_215997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        float_215998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 21), list_215997, float_215998)
        # Adding element type (line 59)
        float_215999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 21), list_215997, float_215999)
        # Adding element type (line 59)
        int_216000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 35), 'int')
        float_216001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'float')
        # Applying the binary operator '-' (line 59)
        result_sub_216002 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 35), '-', int_216000, float_216001)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 21), list_215997, result_sub_216002)
        
        # Processing the call keyword arguments (line 59)
        kwargs_216003 = {}
        # Getting the type of 'np' (line 59)
        np_215995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 59)
        array_215996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), np_215995, 'array')
        # Calling array(args, kwargs) (line 59)
        array_call_result_216004 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), array_215996, *[list_215997], **kwargs_216003)
        
        # Assigning a type to the variable 'x' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'x', array_call_result_216004)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to find_active_constraints(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'x' (line 60)
        x_216006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 41), 'x', False)
        # Getting the type of 'lb' (line 60)
        lb_216007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'lb', False)
        # Getting the type of 'ub' (line 60)
        ub_216008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 48), 'ub', False)
        # Processing the call keyword arguments (line 60)
        kwargs_216009 = {}
        # Getting the type of 'find_active_constraints' (line 60)
        find_active_constraints_216005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'find_active_constraints', False)
        # Calling find_active_constraints(args, kwargs) (line 60)
        find_active_constraints_call_result_216010 = invoke(stypy.reporting.localization.Localization(__file__, 60, 17), find_active_constraints_216005, *[x_216006, lb_216007, ub_216008], **kwargs_216009)
        
        # Assigning a type to the variable 'active' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'active', find_active_constraints_call_result_216010)
        
        # Call to assert_equal(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'active' (line 61)
        active_216012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'active', False)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_216013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_216014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 29), list_216013, int_216014)
        # Adding element type (line 61)
        int_216015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 29), list_216013, int_216015)
        # Adding element type (line 61)
        int_216016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 29), list_216013, int_216016)
        
        # Processing the call keyword arguments (line 61)
        kwargs_216017 = {}
        # Getting the type of 'assert_equal' (line 61)
        assert_equal_216011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 61)
        assert_equal_call_result_216018 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_equal_216011, *[active_216012, list_216013], **kwargs_216017)
        
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to find_active_constraints(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'x' (line 63)
        x_216020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 41), 'x', False)
        # Getting the type of 'lb' (line 63)
        lb_216021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 44), 'lb', False)
        # Getting the type of 'ub' (line 63)
        ub_216022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 48), 'ub', False)
        # Processing the call keyword arguments (line 63)
        float_216023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 57), 'float')
        keyword_216024 = float_216023
        kwargs_216025 = {'rtol': keyword_216024}
        # Getting the type of 'find_active_constraints' (line 63)
        find_active_constraints_216019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'find_active_constraints', False)
        # Calling find_active_constraints(args, kwargs) (line 63)
        find_active_constraints_call_result_216026 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), find_active_constraints_216019, *[x_216020, lb_216021, ub_216022], **kwargs_216025)
        
        # Assigning a type to the variable 'active' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'active', find_active_constraints_call_result_216026)
        
        # Call to assert_equal(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'active' (line 64)
        active_216028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'active', False)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_216029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_216030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 29), list_216029, int_216030)
        # Adding element type (line 64)
        int_216031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 29), list_216029, int_216031)
        # Adding element type (line 64)
        int_216032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 29), list_216029, int_216032)
        
        # Processing the call keyword arguments (line 64)
        kwargs_216033 = {}
        # Getting the type of 'assert_equal' (line 64)
        assert_equal_216027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 64)
        assert_equal_call_result_216034 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_equal_216027, *[active_216028, list_216029], **kwargs_216033)
        
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to array(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_216037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        float_216038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 22), list_216037, float_216038)
        # Adding element type (line 66)
        
        # Getting the type of 'np' (line 66)
        np_216039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'np', False)
        # Obtaining the member 'inf' of a type (line 66)
        inf_216040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 29), np_216039, 'inf')
        # Applying the 'usub' unary operator (line 66)
        result___neg___216041 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 28), 'usub', inf_216040)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 22), list_216037, result___neg___216041)
        # Adding element type (line 66)
        
        # Getting the type of 'np' (line 66)
        np_216042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'np', False)
        # Obtaining the member 'inf' of a type (line 66)
        inf_216043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 38), np_216042, 'inf')
        # Applying the 'usub' unary operator (line 66)
        result___neg___216044 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 37), 'usub', inf_216043)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 22), list_216037, result___neg___216044)
        
        # Processing the call keyword arguments (line 66)
        kwargs_216045 = {}
        # Getting the type of 'np' (line 66)
        np_216035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 66)
        array_216036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 13), np_216035, 'array')
        # Calling array(args, kwargs) (line 66)
        array_call_result_216046 = invoke(stypy.reporting.localization.Localization(__file__, 66, 13), array_216036, *[list_216037], **kwargs_216045)
        
        # Assigning a type to the variable 'lb' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'lb', array_call_result_216046)
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to array(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_216049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        # Getting the type of 'np' (line 67)
        np_216050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'np', False)
        # Obtaining the member 'inf' of a type (line 67)
        inf_216051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 23), np_216050, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 22), list_216049, inf_216051)
        # Adding element type (line 67)
        float_216052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 22), list_216049, float_216052)
        # Adding element type (line 67)
        # Getting the type of 'np' (line 67)
        np_216053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'np', False)
        # Obtaining the member 'inf' of a type (line 67)
        inf_216054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 37), np_216053, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 22), list_216049, inf_216054)
        
        # Processing the call keyword arguments (line 67)
        kwargs_216055 = {}
        # Getting the type of 'np' (line 67)
        np_216047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 67)
        array_216048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 13), np_216047, 'array')
        # Calling array(args, kwargs) (line 67)
        array_call_result_216056 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), array_216048, *[list_216049], **kwargs_216055)
        
        # Assigning a type to the variable 'ub' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'ub', array_call_result_216056)
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to ones(...): (line 69)
        # Processing the call arguments (line 69)
        int_216059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_216060 = {}
        # Getting the type of 'np' (line 69)
        np_216057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 69)
        ones_216058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), np_216057, 'ones')
        # Calling ones(args, kwargs) (line 69)
        ones_call_result_216061 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), ones_216058, *[int_216059], **kwargs_216060)
        
        # Assigning a type to the variable 'x' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'x', ones_call_result_216061)
        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to find_active_constraints(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'x' (line 70)
        x_216063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 41), 'x', False)
        # Getting the type of 'lb' (line 70)
        lb_216064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 44), 'lb', False)
        # Getting the type of 'ub' (line 70)
        ub_216065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 48), 'ub', False)
        # Processing the call keyword arguments (line 70)
        kwargs_216066 = {}
        # Getting the type of 'find_active_constraints' (line 70)
        find_active_constraints_216062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'find_active_constraints', False)
        # Calling find_active_constraints(args, kwargs) (line 70)
        find_active_constraints_call_result_216067 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), find_active_constraints_216062, *[x_216063, lb_216064, ub_216065], **kwargs_216066)
        
        # Assigning a type to the variable 'active' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'active', find_active_constraints_call_result_216067)
        
        # Call to assert_equal(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'active' (line 71)
        active_216069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'active', False)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_216070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        int_216071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 29), list_216070, int_216071)
        # Adding element type (line 71)
        int_216072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 29), list_216070, int_216072)
        # Adding element type (line 71)
        int_216073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 29), list_216070, int_216073)
        
        # Processing the call keyword arguments (line 71)
        kwargs_216074 = {}
        # Getting the type of 'assert_equal' (line 71)
        assert_equal_216068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 71)
        assert_equal_call_result_216075 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert_equal_216068, *[active_216069, list_216070], **kwargs_216074)
        
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to array(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_216078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        float_216079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), list_216078, float_216079)
        # Adding element type (line 74)
        float_216080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), list_216078, float_216080)
        # Adding element type (line 74)
        float_216081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), list_216078, float_216081)
        
        # Processing the call keyword arguments (line 74)
        kwargs_216082 = {}
        # Getting the type of 'np' (line 74)
        np_216076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 74)
        array_216077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), np_216076, 'array')
        # Calling array(args, kwargs) (line 74)
        array_call_result_216083 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), array_216077, *[list_216078], **kwargs_216082)
        
        # Assigning a type to the variable 'x' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'x', array_call_result_216083)
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to find_active_constraints(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'x' (line 75)
        x_216085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 41), 'x', False)
        # Getting the type of 'lb' (line 75)
        lb_216086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 44), 'lb', False)
        # Getting the type of 'ub' (line 75)
        ub_216087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 48), 'ub', False)
        # Processing the call keyword arguments (line 75)
        kwargs_216088 = {}
        # Getting the type of 'find_active_constraints' (line 75)
        find_active_constraints_216084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'find_active_constraints', False)
        # Calling find_active_constraints(args, kwargs) (line 75)
        find_active_constraints_call_result_216089 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), find_active_constraints_216084, *[x_216085, lb_216086, ub_216087], **kwargs_216088)
        
        # Assigning a type to the variable 'active' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'active', find_active_constraints_call_result_216089)
        
        # Call to assert_equal(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'active' (line 76)
        active_216091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 21), 'active', False)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_216092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_216093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_216092, int_216093)
        # Adding element type (line 76)
        int_216094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_216092, int_216094)
        # Adding element type (line 76)
        int_216095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_216092, int_216095)
        
        # Processing the call keyword arguments (line 76)
        kwargs_216096 = {}
        # Getting the type of 'assert_equal' (line 76)
        assert_equal_216090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 76)
        assert_equal_call_result_216097 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_equal_216090, *[active_216091, list_216092], **kwargs_216096)
        
        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to find_active_constraints(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'x' (line 78)
        x_216099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 41), 'x', False)
        # Getting the type of 'lb' (line 78)
        lb_216100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'lb', False)
        # Getting the type of 'ub' (line 78)
        ub_216101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 48), 'ub', False)
        # Processing the call keyword arguments (line 78)
        int_216102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 57), 'int')
        keyword_216103 = int_216102
        kwargs_216104 = {'rtol': keyword_216103}
        # Getting the type of 'find_active_constraints' (line 78)
        find_active_constraints_216098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'find_active_constraints', False)
        # Calling find_active_constraints(args, kwargs) (line 78)
        find_active_constraints_call_result_216105 = invoke(stypy.reporting.localization.Localization(__file__, 78, 17), find_active_constraints_216098, *[x_216099, lb_216100, ub_216101], **kwargs_216104)
        
        # Assigning a type to the variable 'active' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'active', find_active_constraints_call_result_216105)
        
        # Call to assert_equal(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'active' (line 79)
        active_216107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'active', False)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_216108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        int_216109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 29), list_216108, int_216109)
        # Adding element type (line 79)
        int_216110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 29), list_216108, int_216110)
        # Adding element type (line 79)
        int_216111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 29), list_216108, int_216111)
        
        # Processing the call keyword arguments (line 79)
        kwargs_216112 = {}
        # Getting the type of 'assert_equal' (line 79)
        assert_equal_216106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 79)
        assert_equal_call_result_216113 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assert_equal_216106, *[active_216107, list_216108], **kwargs_216112)
        
        
        # ################# End of 'test_find_active_constraints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_find_active_constraints' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_216114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_216114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_find_active_constraints'
        return stypy_return_type_216114


    @norecursion
    def test_make_strictly_feasible(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_strictly_feasible'
        module_type_store = module_type_store.open_function_context('test_make_strictly_feasible', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_localization', localization)
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_function_name', 'TestBounds.test_make_strictly_feasible')
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_param_names_list', [])
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBounds.test_make_strictly_feasible.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBounds.test_make_strictly_feasible', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_strictly_feasible', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_strictly_feasible(...)' code ##################

        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to array(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_216117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        float_216118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_216117, float_216118)
        # Adding element type (line 82)
        float_216119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_216117, float_216119)
        # Adding element type (line 82)
        float_216120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_216117, float_216120)
        
        # Processing the call keyword arguments (line 82)
        kwargs_216121 = {}
        # Getting the type of 'np' (line 82)
        np_216115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 82)
        array_216116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), np_216115, 'array')
        # Calling array(args, kwargs) (line 82)
        array_call_result_216122 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), array_216116, *[list_216117], **kwargs_216121)
        
        # Assigning a type to the variable 'lb' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'lb', array_call_result_216122)
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to array(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_216125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        float_216126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_216125, float_216126)
        # Adding element type (line 83)
        float_216127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_216125, float_216127)
        # Adding element type (line 83)
        float_216128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_216125, float_216128)
        
        # Processing the call keyword arguments (line 83)
        kwargs_216129 = {}
        # Getting the type of 'np' (line 83)
        np_216123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 83)
        array_216124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), np_216123, 'array')
        # Calling array(args, kwargs) (line 83)
        array_call_result_216130 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), array_216124, *[list_216125], **kwargs_216129)
        
        # Assigning a type to the variable 'ub' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'ub', array_call_result_216130)
        
        # Assigning a Call to a Name (line 85):
        
        # Assigning a Call to a Name (line 85):
        
        # Call to array(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_216133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        float_216134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_216133, float_216134)
        # Adding element type (line 85)
        float_216135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_216133, float_216135)
        # Adding element type (line 85)
        int_216136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 33), 'int')
        float_216137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'float')
        # Applying the binary operator '+' (line 85)
        result_add_216138 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 33), '+', int_216136, float_216137)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_216133, result_add_216138)
        
        # Processing the call keyword arguments (line 85)
        kwargs_216139 = {}
        # Getting the type of 'np' (line 85)
        np_216131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 85)
        array_216132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), np_216131, 'array')
        # Calling array(args, kwargs) (line 85)
        array_call_result_216140 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), array_216132, *[list_216133], **kwargs_216139)
        
        # Assigning a type to the variable 'x' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'x', array_call_result_216140)
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to make_strictly_feasible(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'x' (line 87)
        x_216142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 39), 'x', False)
        # Getting the type of 'lb' (line 87)
        lb_216143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'lb', False)
        # Getting the type of 'ub' (line 87)
        ub_216144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 46), 'ub', False)
        # Processing the call keyword arguments (line 87)
        int_216145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 56), 'int')
        keyword_216146 = int_216145
        kwargs_216147 = {'rstep': keyword_216146}
        # Getting the type of 'make_strictly_feasible' (line 87)
        make_strictly_feasible_216141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'make_strictly_feasible', False)
        # Calling make_strictly_feasible(args, kwargs) (line 87)
        make_strictly_feasible_call_result_216148 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), make_strictly_feasible_216141, *[x_216142, lb_216143, ub_216144], **kwargs_216147)
        
        # Assigning a type to the variable 'x_new' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'x_new', make_strictly_feasible_call_result_216148)
        
        # Call to assert_(...): (line 88)
        # Processing the call arguments (line 88)
        
        
        # Obtaining the type of the subscript
        int_216150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'int')
        # Getting the type of 'x_new' (line 88)
        x_new_216151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'x_new', False)
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___216152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), x_new_216151, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_216153 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), getitem___216152, int_216150)
        
        float_216154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 27), 'float')
        # Applying the binary operator '>' (line 88)
        result_gt_216155 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 16), '>', subscript_call_result_216153, float_216154)
        
        # Processing the call keyword arguments (line 88)
        kwargs_216156 = {}
        # Getting the type of 'assert_' (line 88)
        assert__216149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 88)
        assert__call_result_216157 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), assert__216149, *[result_gt_216155], **kwargs_216156)
        
        
        # Call to assert_equal(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining the type of the subscript
        int_216159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 27), 'int')
        slice_216160 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 21), int_216159, None, None)
        # Getting the type of 'x_new' (line 89)
        x_new_216161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'x_new', False)
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___216162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 21), x_new_216161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_216163 = invoke(stypy.reporting.localization.Localization(__file__, 89, 21), getitem___216162, slice_216160)
        
        
        # Obtaining the type of the subscript
        int_216164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 34), 'int')
        slice_216165 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 32), int_216164, None, None)
        # Getting the type of 'x' (line 89)
        x_216166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 32), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___216167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 32), x_216166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_216168 = invoke(stypy.reporting.localization.Localization(__file__, 89, 32), getitem___216167, slice_216165)
        
        # Processing the call keyword arguments (line 89)
        kwargs_216169 = {}
        # Getting the type of 'assert_equal' (line 89)
        assert_equal_216158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 89)
        assert_equal_call_result_216170 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert_equal_216158, *[subscript_call_result_216163, subscript_call_result_216168], **kwargs_216169)
        
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to make_strictly_feasible(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'x' (line 91)
        x_216172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 39), 'x', False)
        # Getting the type of 'lb' (line 91)
        lb_216173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 42), 'lb', False)
        # Getting the type of 'ub' (line 91)
        ub_216174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 46), 'ub', False)
        # Processing the call keyword arguments (line 91)
        float_216175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 56), 'float')
        keyword_216176 = float_216175
        kwargs_216177 = {'rstep': keyword_216176}
        # Getting the type of 'make_strictly_feasible' (line 91)
        make_strictly_feasible_216171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'make_strictly_feasible', False)
        # Calling make_strictly_feasible(args, kwargs) (line 91)
        make_strictly_feasible_call_result_216178 = invoke(stypy.reporting.localization.Localization(__file__, 91, 16), make_strictly_feasible_216171, *[x_216172, lb_216173, ub_216174], **kwargs_216177)
        
        # Assigning a type to the variable 'x_new' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'x_new', make_strictly_feasible_call_result_216178)
        
        # Call to assert_equal(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'x_new' (line 92)
        x_new_216180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 21), 'x_new', False)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_216181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        float_216182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 29), 'float')
        float_216183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 36), 'float')
        # Applying the binary operator '+' (line 92)
        result_add_216184 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 29), '+', float_216182, float_216183)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_216181, result_add_216184)
        # Adding element type (line 92)
        float_216185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_216181, float_216185)
        # Adding element type (line 92)
        int_216186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 47), 'int')
        int_216187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 52), 'int')
        float_216188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 56), 'float')
        # Applying the binary operator '+' (line 92)
        result_add_216189 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 52), '+', int_216187, float_216188)
        
        # Applying the binary operator '*' (line 92)
        result_mul_216190 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 47), '*', int_216186, result_add_216189)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_216181, result_mul_216190)
        
        # Processing the call keyword arguments (line 92)
        kwargs_216191 = {}
        # Getting the type of 'assert_equal' (line 92)
        assert_equal_216179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 92)
        assert_equal_call_result_216192 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), assert_equal_216179, *[x_new_216180, list_216181], **kwargs_216191)
        
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to array(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_216195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        float_216196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_216195, float_216196)
        # Adding element type (line 94)
        int_216197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_216195, int_216197)
        # Adding element type (line 94)
        float_216198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_216195, float_216198)
        
        # Processing the call keyword arguments (line 94)
        kwargs_216199 = {}
        # Getting the type of 'np' (line 94)
        np_216193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 94)
        array_216194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), np_216193, 'array')
        # Calling array(args, kwargs) (line 94)
        array_call_result_216200 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), array_216194, *[list_216195], **kwargs_216199)
        
        # Assigning a type to the variable 'x' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'x', array_call_result_216200)
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to make_strictly_feasible(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'x' (line 95)
        x_216202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 39), 'x', False)
        # Getting the type of 'lb' (line 95)
        lb_216203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 42), 'lb', False)
        # Getting the type of 'ub' (line 95)
        ub_216204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 46), 'ub', False)
        # Processing the call keyword arguments (line 95)
        kwargs_216205 = {}
        # Getting the type of 'make_strictly_feasible' (line 95)
        make_strictly_feasible_216201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'make_strictly_feasible', False)
        # Calling make_strictly_feasible(args, kwargs) (line 95)
        make_strictly_feasible_call_result_216206 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), make_strictly_feasible_216201, *[x_216202, lb_216203, ub_216204], **kwargs_216205)
        
        # Assigning a type to the variable 'x_new' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'x_new', make_strictly_feasible_call_result_216206)
        
        # Call to assert_(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to all(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Getting the type of 'x_new' (line 96)
        x_new_216210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'x_new', False)
        # Getting the type of 'lb' (line 96)
        lb_216211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'lb', False)
        # Applying the binary operator '>=' (line 96)
        result_ge_216212 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 24), '>=', x_new_216210, lb_216211)
        
        
        # Getting the type of 'x_new' (line 96)
        x_new_216213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 40), 'x_new', False)
        # Getting the type of 'ub' (line 96)
        ub_216214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 49), 'ub', False)
        # Applying the binary operator '<=' (line 96)
        result_le_216215 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 40), '<=', x_new_216213, ub_216214)
        
        # Applying the binary operator '&' (line 96)
        result_and__216216 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 23), '&', result_ge_216212, result_le_216215)
        
        # Processing the call keyword arguments (line 96)
        kwargs_216217 = {}
        # Getting the type of 'np' (line 96)
        np_216208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 96)
        all_216209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), np_216208, 'all')
        # Calling all(args, kwargs) (line 96)
        all_call_result_216218 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), all_216209, *[result_and__216216], **kwargs_216217)
        
        # Processing the call keyword arguments (line 96)
        kwargs_216219 = {}
        # Getting the type of 'assert_' (line 96)
        assert__216207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 96)
        assert__call_result_216220 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assert__216207, *[all_call_result_216218], **kwargs_216219)
        
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to make_strictly_feasible(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x' (line 98)
        x_216222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 39), 'x', False)
        # Getting the type of 'lb' (line 98)
        lb_216223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'lb', False)
        # Getting the type of 'ub' (line 98)
        ub_216224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 46), 'ub', False)
        # Processing the call keyword arguments (line 98)
        int_216225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 56), 'int')
        keyword_216226 = int_216225
        kwargs_216227 = {'rstep': keyword_216226}
        # Getting the type of 'make_strictly_feasible' (line 98)
        make_strictly_feasible_216221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'make_strictly_feasible', False)
        # Calling make_strictly_feasible(args, kwargs) (line 98)
        make_strictly_feasible_call_result_216228 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), make_strictly_feasible_216221, *[x_216222, lb_216223, ub_216224], **kwargs_216227)
        
        # Assigning a type to the variable 'x_new' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'x_new', make_strictly_feasible_call_result_216228)
        
        # Call to assert_(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to all(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Getting the type of 'x_new' (line 99)
        x_new_216232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'x_new', False)
        # Getting the type of 'lb' (line 99)
        lb_216233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'lb', False)
        # Applying the binary operator '>=' (line 99)
        result_ge_216234 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 24), '>=', x_new_216232, lb_216233)
        
        
        # Getting the type of 'x_new' (line 99)
        x_new_216235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'x_new', False)
        # Getting the type of 'ub' (line 99)
        ub_216236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 49), 'ub', False)
        # Applying the binary operator '<=' (line 99)
        result_le_216237 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 40), '<=', x_new_216235, ub_216236)
        
        # Applying the binary operator '&' (line 99)
        result_and__216238 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 23), '&', result_ge_216234, result_le_216237)
        
        # Processing the call keyword arguments (line 99)
        kwargs_216239 = {}
        # Getting the type of 'np' (line 99)
        np_216230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 99)
        all_216231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), np_216230, 'all')
        # Calling all(args, kwargs) (line 99)
        all_call_result_216240 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), all_216231, *[result_and__216238], **kwargs_216239)
        
        # Processing the call keyword arguments (line 99)
        kwargs_216241 = {}
        # Getting the type of 'assert_' (line 99)
        assert__216229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 99)
        assert__call_result_216242 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assert__216229, *[all_call_result_216240], **kwargs_216241)
        
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to array(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_216245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        int_216246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_216245, int_216246)
        # Adding element type (line 101)
        float_216247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_216245, float_216247)
        
        # Processing the call keyword arguments (line 101)
        kwargs_216248 = {}
        # Getting the type of 'np' (line 101)
        np_216243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 101)
        array_216244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), np_216243, 'array')
        # Calling array(args, kwargs) (line 101)
        array_call_result_216249 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), array_216244, *[list_216245], **kwargs_216248)
        
        # Assigning a type to the variable 'lb' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'lb', array_call_result_216249)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to array(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Obtaining an instance of the builtin type 'list' (line 102)
        list_216252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 102)
        # Adding element type (line 102)
        int_216253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 22), list_216252, int_216253)
        # Adding element type (line 102)
        float_216254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 26), 'float')
        float_216255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 34), 'float')
        # Applying the binary operator '+' (line 102)
        result_add_216256 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 26), '+', float_216254, float_216255)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 22), list_216252, result_add_216256)
        
        # Processing the call keyword arguments (line 102)
        kwargs_216257 = {}
        # Getting the type of 'np' (line 102)
        np_216250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 102)
        array_216251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), np_216250, 'array')
        # Calling array(args, kwargs) (line 102)
        array_call_result_216258 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), array_216251, *[list_216252], **kwargs_216257)
        
        # Assigning a type to the variable 'ub' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'ub', array_call_result_216258)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to array(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_216261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        int_216262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 21), list_216261, int_216262)
        # Adding element type (line 103)
        float_216263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 21), list_216261, float_216263)
        
        # Processing the call keyword arguments (line 103)
        kwargs_216264 = {}
        # Getting the type of 'np' (line 103)
        np_216259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 103)
        array_216260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), np_216259, 'array')
        # Calling array(args, kwargs) (line 103)
        array_call_result_216265 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), array_216260, *[list_216261], **kwargs_216264)
        
        # Assigning a type to the variable 'x' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'x', array_call_result_216265)
        
        # Assigning a Call to a Name (line 104):
        
        # Assigning a Call to a Name (line 104):
        
        # Call to make_strictly_feasible(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'x' (line 104)
        x_216267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'x', False)
        # Getting the type of 'lb' (line 104)
        lb_216268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 42), 'lb', False)
        # Getting the type of 'ub' (line 104)
        ub_216269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 46), 'ub', False)
        # Processing the call keyword arguments (line 104)
        float_216270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 56), 'float')
        keyword_216271 = float_216270
        kwargs_216272 = {'rstep': keyword_216271}
        # Getting the type of 'make_strictly_feasible' (line 104)
        make_strictly_feasible_216266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'make_strictly_feasible', False)
        # Calling make_strictly_feasible(args, kwargs) (line 104)
        make_strictly_feasible_call_result_216273 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), make_strictly_feasible_216266, *[x_216267, lb_216268, ub_216269], **kwargs_216272)
        
        # Assigning a type to the variable 'x_new' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'x_new', make_strictly_feasible_call_result_216273)
        
        # Call to assert_equal(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'x_new' (line 105)
        x_new_216275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'x_new', False)
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_216276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        int_216277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 28), list_216276, int_216277)
        # Adding element type (line 105)
        float_216278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 32), 'float')
        float_216279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 40), 'float')
        # Applying the binary operator '+' (line 105)
        result_add_216280 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 32), '+', float_216278, float_216279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 28), list_216276, result_add_216280)
        
        # Processing the call keyword arguments (line 105)
        kwargs_216281 = {}
        # Getting the type of 'assert_equal' (line 105)
        assert_equal_216274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 105)
        assert_equal_call_result_216282 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assert_equal_216274, *[x_new_216275, list_216276], **kwargs_216281)
        
        
        # ################# End of 'test_make_strictly_feasible(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_strictly_feasible' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_216283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_216283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_strictly_feasible'
        return stypy_return_type_216283


    @norecursion
    def test_scaling_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scaling_vector'
        module_type_store = module_type_store.open_function_context('test_scaling_vector', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_localization', localization)
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_function_name', 'TestBounds.test_scaling_vector')
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_param_names_list', [])
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBounds.test_scaling_vector.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBounds.test_scaling_vector', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scaling_vector', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scaling_vector(...)' code ##################

        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to array(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_216286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        
        # Getting the type of 'np' (line 108)
        np_216287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'np', False)
        # Obtaining the member 'inf' of a type (line 108)
        inf_216288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 24), np_216287, 'inf')
        # Applying the 'usub' unary operator (line 108)
        result___neg___216289 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 23), 'usub', inf_216288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_216286, result___neg___216289)
        # Adding element type (line 108)
        float_216290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_216286, float_216290)
        # Adding element type (line 108)
        float_216291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_216286, float_216291)
        # Adding element type (line 108)
        
        # Getting the type of 'np' (line 108)
        np_216292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'np', False)
        # Obtaining the member 'inf' of a type (line 108)
        inf_216293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 44), np_216292, 'inf')
        # Applying the 'usub' unary operator (line 108)
        result___neg___216294 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 43), 'usub', inf_216293)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_216286, result___neg___216294)
        
        # Processing the call keyword arguments (line 108)
        kwargs_216295 = {}
        # Getting the type of 'np' (line 108)
        np_216284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 108)
        array_216285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), np_216284, 'array')
        # Calling array(args, kwargs) (line 108)
        array_call_result_216296 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), array_216285, *[list_216286], **kwargs_216295)
        
        # Assigning a type to the variable 'lb' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'lb', array_call_result_216296)
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to array(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_216299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        float_216300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_216299, float_216300)
        # Adding element type (line 109)
        # Getting the type of 'np' (line 109)
        np_216301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'np', False)
        # Obtaining the member 'inf' of a type (line 109)
        inf_216302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 28), np_216301, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_216299, inf_216302)
        # Adding element type (line 109)
        float_216303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_216299, float_216303)
        # Adding element type (line 109)
        # Getting the type of 'np' (line 109)
        np_216304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 42), 'np', False)
        # Obtaining the member 'inf' of a type (line 109)
        inf_216305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 42), np_216304, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_216299, inf_216305)
        
        # Processing the call keyword arguments (line 109)
        kwargs_216306 = {}
        # Getting the type of 'np' (line 109)
        np_216297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 109)
        array_216298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), np_216297, 'array')
        # Calling array(args, kwargs) (line 109)
        array_call_result_216307 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), array_216298, *[list_216299], **kwargs_216306)
        
        # Assigning a type to the variable 'ub' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'ub', array_call_result_216307)
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to array(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_216310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        float_216311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_216310, float_216311)
        # Adding element type (line 110)
        float_216312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_216310, float_216312)
        # Adding element type (line 110)
        float_216313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_216310, float_216313)
        # Adding element type (line 110)
        float_216314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_216310, float_216314)
        
        # Processing the call keyword arguments (line 110)
        kwargs_216315 = {}
        # Getting the type of 'np' (line 110)
        np_216308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 110)
        array_216309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), np_216308, 'array')
        # Calling array(args, kwargs) (line 110)
        array_call_result_216316 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), array_216309, *[list_216310], **kwargs_216315)
        
        # Assigning a type to the variable 'x' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'x', array_call_result_216316)
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to array(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_216319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        float_216320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_216319, float_216320)
        # Adding element type (line 111)
        float_216321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_216319, float_216321)
        # Adding element type (line 111)
        float_216322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_216319, float_216322)
        # Adding element type (line 111)
        float_216323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_216319, float_216323)
        
        # Processing the call keyword arguments (line 111)
        kwargs_216324 = {}
        # Getting the type of 'np' (line 111)
        np_216317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 111)
        array_216318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), np_216317, 'array')
        # Calling array(args, kwargs) (line 111)
        array_call_result_216325 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), array_216318, *[list_216319], **kwargs_216324)
        
        # Assigning a type to the variable 'g' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'g', array_call_result_216325)
        
        # Assigning a Call to a Tuple (line 112):
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_216326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        
        # Call to CL_scaling_vector(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'x' (line 112)
        x_216328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'x', False)
        # Getting the type of 'g' (line 112)
        g_216329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 37), 'g', False)
        # Getting the type of 'lb' (line 112)
        lb_216330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'lb', False)
        # Getting the type of 'ub' (line 112)
        ub_216331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 44), 'ub', False)
        # Processing the call keyword arguments (line 112)
        kwargs_216332 = {}
        # Getting the type of 'CL_scaling_vector' (line 112)
        CL_scaling_vector_216327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'CL_scaling_vector', False)
        # Calling CL_scaling_vector(args, kwargs) (line 112)
        CL_scaling_vector_call_result_216333 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), CL_scaling_vector_216327, *[x_216328, g_216329, lb_216330, ub_216331], **kwargs_216332)
        
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___216334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), CL_scaling_vector_call_result_216333, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_216335 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), getitem___216334, int_216326)
        
        # Assigning a type to the variable 'tuple_var_assignment_215632' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_215632', subscript_call_result_216335)
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_216336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        
        # Call to CL_scaling_vector(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'x' (line 112)
        x_216338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'x', False)
        # Getting the type of 'g' (line 112)
        g_216339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 37), 'g', False)
        # Getting the type of 'lb' (line 112)
        lb_216340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'lb', False)
        # Getting the type of 'ub' (line 112)
        ub_216341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 44), 'ub', False)
        # Processing the call keyword arguments (line 112)
        kwargs_216342 = {}
        # Getting the type of 'CL_scaling_vector' (line 112)
        CL_scaling_vector_216337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'CL_scaling_vector', False)
        # Calling CL_scaling_vector(args, kwargs) (line 112)
        CL_scaling_vector_call_result_216343 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), CL_scaling_vector_216337, *[x_216338, g_216339, lb_216340, ub_216341], **kwargs_216342)
        
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___216344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), CL_scaling_vector_call_result_216343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_216345 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), getitem___216344, int_216336)
        
        # Assigning a type to the variable 'tuple_var_assignment_215633' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_215633', subscript_call_result_216345)
        
        # Assigning a Name to a Name (line 112):
        # Getting the type of 'tuple_var_assignment_215632' (line 112)
        tuple_var_assignment_215632_216346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_215632')
        # Assigning a type to the variable 'v' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'v', tuple_var_assignment_215632_216346)
        
        # Assigning a Name to a Name (line 112):
        # Getting the type of 'tuple_var_assignment_215633' (line 112)
        tuple_var_assignment_215633_216347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_215633')
        # Assigning a type to the variable 'dv' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'dv', tuple_var_assignment_215633_216347)
        
        # Call to assert_equal(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'v' (line 113)
        v_216349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'v', False)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_216350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        float_216351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_216350, float_216351)
        # Adding element type (line 113)
        float_216352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_216350, float_216352)
        # Adding element type (line 113)
        float_216353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_216350, float_216353)
        # Adding element type (line 113)
        float_216354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_216350, float_216354)
        
        # Processing the call keyword arguments (line 113)
        kwargs_216355 = {}
        # Getting the type of 'assert_equal' (line 113)
        assert_equal_216348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 113)
        assert_equal_call_result_216356 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_equal_216348, *[v_216349, list_216350], **kwargs_216355)
        
        
        # Call to assert_equal(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'dv' (line 114)
        dv_216358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'dv', False)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_216359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        float_216360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 25), list_216359, float_216360)
        # Adding element type (line 114)
        float_216361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 25), list_216359, float_216361)
        # Adding element type (line 114)
        float_216362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 25), list_216359, float_216362)
        # Adding element type (line 114)
        float_216363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 25), list_216359, float_216363)
        
        # Processing the call keyword arguments (line 114)
        kwargs_216364 = {}
        # Getting the type of 'assert_equal' (line 114)
        assert_equal_216357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 114)
        assert_equal_call_result_216365 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assert_equal_216357, *[dv_216358, list_216359], **kwargs_216364)
        
        
        # ################# End of 'test_scaling_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scaling_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_216366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_216366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scaling_vector'
        return stypy_return_type_216366


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 0, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBounds.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBounds' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'TestBounds', TestBounds)
# Declaration of the 'TestQuadraticFunction' class

class TestQuadraticFunction(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_function_name', 'TestQuadraticFunction.setup_method')
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadraticFunction.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadraticFunction.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 119):
        
        # Assigning a Call to a Attribute (line 119):
        
        # Call to array(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_216369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_216370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        float_216371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 12), list_216370, float_216371)
        # Adding element type (line 120)
        float_216372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 12), list_216370, float_216372)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_216369, list_216370)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_216373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        float_216374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 12), list_216373, float_216374)
        # Adding element type (line 121)
        float_216375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 12), list_216373, float_216375)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_216369, list_216373)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_216376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        float_216377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 12), list_216376, float_216377)
        # Adding element type (line 122)
        float_216378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 12), list_216376, float_216378)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_216369, list_216376)
        
        # Processing the call keyword arguments (line 119)
        kwargs_216379 = {}
        # Getting the type of 'np' (line 119)
        np_216367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 119)
        array_216368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 17), np_216367, 'array')
        # Calling array(args, kwargs) (line 119)
        array_call_result_216380 = invoke(stypy.reporting.localization.Localization(__file__, 119, 17), array_216368, *[list_216369], **kwargs_216379)
        
        # Getting the type of 'self' (line 119)
        self_216381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'J' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_216381, 'J', array_call_result_216380)
        
        # Assigning a Call to a Attribute (line 123):
        
        # Assigning a Call to a Attribute (line 123):
        
        # Call to array(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_216384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        float_216385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 26), list_216384, float_216385)
        # Adding element type (line 123)
        float_216386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 26), list_216384, float_216386)
        
        # Processing the call keyword arguments (line 123)
        kwargs_216387 = {}
        # Getting the type of 'np' (line 123)
        np_216382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 123)
        array_216383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 17), np_216382, 'array')
        # Calling array(args, kwargs) (line 123)
        array_call_result_216388 = invoke(stypy.reporting.localization.Localization(__file__, 123, 17), array_216383, *[list_216384], **kwargs_216387)
        
        # Getting the type of 'self' (line 123)
        self_216389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self')
        # Setting the type of the member 'g' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_216389, 'g', array_call_result_216388)
        
        # Assigning a Call to a Attribute (line 124):
        
        # Assigning a Call to a Attribute (line 124):
        
        # Call to array(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_216392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        float_216393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 29), list_216392, float_216393)
        # Adding element type (line 124)
        float_216394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 29), list_216392, float_216394)
        
        # Processing the call keyword arguments (line 124)
        kwargs_216395 = {}
        # Getting the type of 'np' (line 124)
        np_216390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 124)
        array_216391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), np_216390, 'array')
        # Calling array(args, kwargs) (line 124)
        array_call_result_216396 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), array_216391, *[list_216392], **kwargs_216395)
        
        # Getting the type of 'self' (line 124)
        self_216397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self')
        # Setting the type of the member 'diag' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_216397, 'diag', array_call_result_216396)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_216398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_216398)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_216398


    @norecursion
    def test_build_quadratic_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_build_quadratic_1d'
        module_type_store = module_type_store.open_function_context('test_build_quadratic_1d', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_localization', localization)
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_function_name', 'TestQuadraticFunction.test_build_quadratic_1d')
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadraticFunction.test_build_quadratic_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadraticFunction.test_build_quadratic_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_build_quadratic_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_build_quadratic_1d(...)' code ##################

        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to zeros(...): (line 127)
        # Processing the call arguments (line 127)
        int_216401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 21), 'int')
        # Processing the call keyword arguments (line 127)
        kwargs_216402 = {}
        # Getting the type of 'np' (line 127)
        np_216399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 127)
        zeros_216400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), np_216399, 'zeros')
        # Calling zeros(args, kwargs) (line 127)
        zeros_call_result_216403 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), zeros_216400, *[int_216401], **kwargs_216402)
        
        # Assigning a type to the variable 's' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 's', zeros_call_result_216403)
        
        # Assigning a Call to a Tuple (line 128):
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_216404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'self' (line 128)
        self_216406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'self', False)
        # Obtaining the member 'J' of a type (line 128)
        J_216407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 34), self_216406, 'J')
        # Getting the type of 'self' (line 128)
        self_216408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'self', False)
        # Obtaining the member 'g' of a type (line 128)
        g_216409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 42), self_216408, 'g')
        # Getting the type of 's' (line 128)
        s_216410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 's', False)
        # Processing the call keyword arguments (line 128)
        kwargs_216411 = {}
        # Getting the type of 'build_quadratic_1d' (line 128)
        build_quadratic_1d_216405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 128)
        build_quadratic_1d_call_result_216412 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), build_quadratic_1d_216405, *[J_216407, g_216409, s_216410], **kwargs_216411)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___216413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), build_quadratic_1d_call_result_216412, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_216414 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___216413, int_216404)
        
        # Assigning a type to the variable 'tuple_var_assignment_215634' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_215634', subscript_call_result_216414)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_216415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'self' (line 128)
        self_216417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'self', False)
        # Obtaining the member 'J' of a type (line 128)
        J_216418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 34), self_216417, 'J')
        # Getting the type of 'self' (line 128)
        self_216419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'self', False)
        # Obtaining the member 'g' of a type (line 128)
        g_216420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 42), self_216419, 'g')
        # Getting the type of 's' (line 128)
        s_216421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 's', False)
        # Processing the call keyword arguments (line 128)
        kwargs_216422 = {}
        # Getting the type of 'build_quadratic_1d' (line 128)
        build_quadratic_1d_216416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 128)
        build_quadratic_1d_call_result_216423 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), build_quadratic_1d_216416, *[J_216418, g_216420, s_216421], **kwargs_216422)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___216424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), build_quadratic_1d_call_result_216423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_216425 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___216424, int_216415)
        
        # Assigning a type to the variable 'tuple_var_assignment_215635' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_215635', subscript_call_result_216425)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_215634' (line 128)
        tuple_var_assignment_215634_216426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_215634')
        # Assigning a type to the variable 'a' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'a', tuple_var_assignment_215634_216426)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_215635' (line 128)
        tuple_var_assignment_215635_216427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_215635')
        # Assigning a type to the variable 'b' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'b', tuple_var_assignment_215635_216427)
        
        # Call to assert_equal(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'a' (line 129)
        a_216429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'a', False)
        int_216430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'int')
        # Processing the call keyword arguments (line 129)
        kwargs_216431 = {}
        # Getting the type of 'assert_equal' (line 129)
        assert_equal_216428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 129)
        assert_equal_call_result_216432 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assert_equal_216428, *[a_216429, int_216430], **kwargs_216431)
        
        
        # Call to assert_equal(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'b' (line 130)
        b_216434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'b', False)
        int_216435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 24), 'int')
        # Processing the call keyword arguments (line 130)
        kwargs_216436 = {}
        # Getting the type of 'assert_equal' (line 130)
        assert_equal_216433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 130)
        assert_equal_call_result_216437 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assert_equal_216433, *[b_216434, int_216435], **kwargs_216436)
        
        
        # Assigning a Call to a Tuple (line 132):
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        int_216438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'self' (line 132)
        self_216440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'self', False)
        # Obtaining the member 'J' of a type (line 132)
        J_216441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 34), self_216440, 'J')
        # Getting the type of 'self' (line 132)
        self_216442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'self', False)
        # Obtaining the member 'g' of a type (line 132)
        g_216443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 42), self_216442, 'g')
        # Getting the type of 's' (line 132)
        s_216444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 50), 's', False)
        # Processing the call keyword arguments (line 132)
        # Getting the type of 'self' (line 132)
        self_216445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 58), 'self', False)
        # Obtaining the member 'diag' of a type (line 132)
        diag_216446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 58), self_216445, 'diag')
        keyword_216447 = diag_216446
        kwargs_216448 = {'diag': keyword_216447}
        # Getting the type of 'build_quadratic_1d' (line 132)
        build_quadratic_1d_216439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 132)
        build_quadratic_1d_call_result_216449 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), build_quadratic_1d_216439, *[J_216441, g_216443, s_216444], **kwargs_216448)
        
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___216450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), build_quadratic_1d_call_result_216449, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_216451 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), getitem___216450, int_216438)
        
        # Assigning a type to the variable 'tuple_var_assignment_215636' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tuple_var_assignment_215636', subscript_call_result_216451)
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        int_216452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'self' (line 132)
        self_216454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'self', False)
        # Obtaining the member 'J' of a type (line 132)
        J_216455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 34), self_216454, 'J')
        # Getting the type of 'self' (line 132)
        self_216456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'self', False)
        # Obtaining the member 'g' of a type (line 132)
        g_216457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 42), self_216456, 'g')
        # Getting the type of 's' (line 132)
        s_216458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 50), 's', False)
        # Processing the call keyword arguments (line 132)
        # Getting the type of 'self' (line 132)
        self_216459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 58), 'self', False)
        # Obtaining the member 'diag' of a type (line 132)
        diag_216460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 58), self_216459, 'diag')
        keyword_216461 = diag_216460
        kwargs_216462 = {'diag': keyword_216461}
        # Getting the type of 'build_quadratic_1d' (line 132)
        build_quadratic_1d_216453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 132)
        build_quadratic_1d_call_result_216463 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), build_quadratic_1d_216453, *[J_216455, g_216457, s_216458], **kwargs_216462)
        
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___216464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), build_quadratic_1d_call_result_216463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_216465 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), getitem___216464, int_216452)
        
        # Assigning a type to the variable 'tuple_var_assignment_215637' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tuple_var_assignment_215637', subscript_call_result_216465)
        
        # Assigning a Name to a Name (line 132):
        # Getting the type of 'tuple_var_assignment_215636' (line 132)
        tuple_var_assignment_215636_216466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tuple_var_assignment_215636')
        # Assigning a type to the variable 'a' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'a', tuple_var_assignment_215636_216466)
        
        # Assigning a Name to a Name (line 132):
        # Getting the type of 'tuple_var_assignment_215637' (line 132)
        tuple_var_assignment_215637_216467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tuple_var_assignment_215637')
        # Assigning a type to the variable 'b' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'b', tuple_var_assignment_215637_216467)
        
        # Call to assert_equal(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'a' (line 133)
        a_216469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 21), 'a', False)
        int_216470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 24), 'int')
        # Processing the call keyword arguments (line 133)
        kwargs_216471 = {}
        # Getting the type of 'assert_equal' (line 133)
        assert_equal_216468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 133)
        assert_equal_call_result_216472 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assert_equal_216468, *[a_216469, int_216470], **kwargs_216471)
        
        
        # Call to assert_equal(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'b' (line 134)
        b_216474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 21), 'b', False)
        int_216475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 24), 'int')
        # Processing the call keyword arguments (line 134)
        kwargs_216476 = {}
        # Getting the type of 'assert_equal' (line 134)
        assert_equal_216473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 134)
        assert_equal_call_result_216477 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), assert_equal_216473, *[b_216474, int_216475], **kwargs_216476)
        
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to array(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_216480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        float_216481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 21), list_216480, float_216481)
        # Adding element type (line 136)
        float_216482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 21), list_216480, float_216482)
        
        # Processing the call keyword arguments (line 136)
        kwargs_216483 = {}
        # Getting the type of 'np' (line 136)
        np_216478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 136)
        array_216479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), np_216478, 'array')
        # Calling array(args, kwargs) (line 136)
        array_call_result_216484 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), array_216479, *[list_216480], **kwargs_216483)
        
        # Assigning a type to the variable 's' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 's', array_call_result_216484)
        
        # Assigning a Call to a Tuple (line 137):
        
        # Assigning a Subscript to a Name (line 137):
        
        # Obtaining the type of the subscript
        int_216485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_216487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'self', False)
        # Obtaining the member 'J' of a type (line 137)
        J_216488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 34), self_216487, 'J')
        # Getting the type of 'self' (line 137)
        self_216489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'self', False)
        # Obtaining the member 'g' of a type (line 137)
        g_216490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 42), self_216489, 'g')
        # Getting the type of 's' (line 137)
        s_216491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 's', False)
        # Processing the call keyword arguments (line 137)
        kwargs_216492 = {}
        # Getting the type of 'build_quadratic_1d' (line 137)
        build_quadratic_1d_216486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 137)
        build_quadratic_1d_call_result_216493 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), build_quadratic_1d_216486, *[J_216488, g_216490, s_216491], **kwargs_216492)
        
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___216494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), build_quadratic_1d_call_result_216493, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_216495 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___216494, int_216485)
        
        # Assigning a type to the variable 'tuple_var_assignment_215638' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_215638', subscript_call_result_216495)
        
        # Assigning a Subscript to a Name (line 137):
        
        # Obtaining the type of the subscript
        int_216496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_216498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'self', False)
        # Obtaining the member 'J' of a type (line 137)
        J_216499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 34), self_216498, 'J')
        # Getting the type of 'self' (line 137)
        self_216500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'self', False)
        # Obtaining the member 'g' of a type (line 137)
        g_216501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 42), self_216500, 'g')
        # Getting the type of 's' (line 137)
        s_216502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 's', False)
        # Processing the call keyword arguments (line 137)
        kwargs_216503 = {}
        # Getting the type of 'build_quadratic_1d' (line 137)
        build_quadratic_1d_216497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 137)
        build_quadratic_1d_call_result_216504 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), build_quadratic_1d_216497, *[J_216499, g_216501, s_216502], **kwargs_216503)
        
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___216505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), build_quadratic_1d_call_result_216504, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_216506 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___216505, int_216496)
        
        # Assigning a type to the variable 'tuple_var_assignment_215639' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_215639', subscript_call_result_216506)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'tuple_var_assignment_215638' (line 137)
        tuple_var_assignment_215638_216507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_215638')
        # Assigning a type to the variable 'a' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'a', tuple_var_assignment_215638_216507)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'tuple_var_assignment_215639' (line 137)
        tuple_var_assignment_215639_216508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_215639')
        # Assigning a type to the variable 'b' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'b', tuple_var_assignment_215639_216508)
        
        # Call to assert_equal(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'a' (line 138)
        a_216510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'a', False)
        float_216511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'float')
        # Processing the call keyword arguments (line 138)
        kwargs_216512 = {}
        # Getting the type of 'assert_equal' (line 138)
        assert_equal_216509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 138)
        assert_equal_call_result_216513 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), assert_equal_216509, *[a_216510, float_216511], **kwargs_216512)
        
        
        # Call to assert_equal(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'b' (line 139)
        b_216515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'b', False)
        float_216516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 24), 'float')
        # Processing the call keyword arguments (line 139)
        kwargs_216517 = {}
        # Getting the type of 'assert_equal' (line 139)
        assert_equal_216514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 139)
        assert_equal_call_result_216518 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assert_equal_216514, *[b_216515, float_216516], **kwargs_216517)
        
        
        # Assigning a Call to a Tuple (line 141):
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_216519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'self' (line 141)
        self_216521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'self', False)
        # Obtaining the member 'J' of a type (line 141)
        J_216522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 34), self_216521, 'J')
        # Getting the type of 'self' (line 141)
        self_216523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), 'self', False)
        # Obtaining the member 'g' of a type (line 141)
        g_216524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 42), self_216523, 'g')
        # Getting the type of 's' (line 141)
        s_216525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 50), 's', False)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'self' (line 141)
        self_216526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 58), 'self', False)
        # Obtaining the member 'diag' of a type (line 141)
        diag_216527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 58), self_216526, 'diag')
        keyword_216528 = diag_216527
        kwargs_216529 = {'diag': keyword_216528}
        # Getting the type of 'build_quadratic_1d' (line 141)
        build_quadratic_1d_216520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 141)
        build_quadratic_1d_call_result_216530 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), build_quadratic_1d_216520, *[J_216522, g_216524, s_216525], **kwargs_216529)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___216531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), build_quadratic_1d_call_result_216530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_216532 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___216531, int_216519)
        
        # Assigning a type to the variable 'tuple_var_assignment_215640' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_215640', subscript_call_result_216532)
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_216533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'self' (line 141)
        self_216535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'self', False)
        # Obtaining the member 'J' of a type (line 141)
        J_216536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 34), self_216535, 'J')
        # Getting the type of 'self' (line 141)
        self_216537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), 'self', False)
        # Obtaining the member 'g' of a type (line 141)
        g_216538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 42), self_216537, 'g')
        # Getting the type of 's' (line 141)
        s_216539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 50), 's', False)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'self' (line 141)
        self_216540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 58), 'self', False)
        # Obtaining the member 'diag' of a type (line 141)
        diag_216541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 58), self_216540, 'diag')
        keyword_216542 = diag_216541
        kwargs_216543 = {'diag': keyword_216542}
        # Getting the type of 'build_quadratic_1d' (line 141)
        build_quadratic_1d_216534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 141)
        build_quadratic_1d_call_result_216544 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), build_quadratic_1d_216534, *[J_216536, g_216538, s_216539], **kwargs_216543)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___216545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), build_quadratic_1d_call_result_216544, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_216546 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___216545, int_216533)
        
        # Assigning a type to the variable 'tuple_var_assignment_215641' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_215641', subscript_call_result_216546)
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_215640' (line 141)
        tuple_var_assignment_215640_216547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_215640')
        # Assigning a type to the variable 'a' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'a', tuple_var_assignment_215640_216547)
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_215641' (line 141)
        tuple_var_assignment_215641_216548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_215641')
        # Assigning a type to the variable 'b' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'b', tuple_var_assignment_215641_216548)
        
        # Call to assert_equal(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'a' (line 142)
        a_216550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'a', False)
        float_216551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 24), 'float')
        # Processing the call keyword arguments (line 142)
        kwargs_216552 = {}
        # Getting the type of 'assert_equal' (line 142)
        assert_equal_216549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 142)
        assert_equal_call_result_216553 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assert_equal_216549, *[a_216550, float_216551], **kwargs_216552)
        
        
        # Call to assert_equal(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'b' (line 143)
        b_216555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'b', False)
        float_216556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 24), 'float')
        # Processing the call keyword arguments (line 143)
        kwargs_216557 = {}
        # Getting the type of 'assert_equal' (line 143)
        assert_equal_216554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 143)
        assert_equal_call_result_216558 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assert_equal_216554, *[b_216555, float_216556], **kwargs_216557)
        
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to array(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_216561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        # Adding element type (line 145)
        float_216562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 22), list_216561, float_216562)
        # Adding element type (line 145)
        float_216563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 22), list_216561, float_216563)
        
        # Processing the call keyword arguments (line 145)
        kwargs_216564 = {}
        # Getting the type of 'np' (line 145)
        np_216559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 145)
        array_216560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 13), np_216559, 'array')
        # Calling array(args, kwargs) (line 145)
        array_call_result_216565 = invoke(stypy.reporting.localization.Localization(__file__, 145, 13), array_216560, *[list_216561], **kwargs_216564)
        
        # Assigning a type to the variable 's0' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 's0', array_call_result_216565)
        
        # Assigning a Call to a Tuple (line 146):
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_216566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_216568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'self', False)
        # Obtaining the member 'J' of a type (line 146)
        J_216569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 37), self_216568, 'J')
        # Getting the type of 'self' (line 146)
        self_216570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 45), 'self', False)
        # Obtaining the member 'g' of a type (line 146)
        g_216571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 45), self_216570, 'g')
        # Getting the type of 's' (line 146)
        s_216572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 53), 's', False)
        # Processing the call keyword arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_216573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 61), 'self', False)
        # Obtaining the member 'diag' of a type (line 146)
        diag_216574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 61), self_216573, 'diag')
        keyword_216575 = diag_216574
        # Getting the type of 's0' (line 146)
        s0_216576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 75), 's0', False)
        keyword_216577 = s0_216576
        kwargs_216578 = {'diag': keyword_216575, 's0': keyword_216577}
        # Getting the type of 'build_quadratic_1d' (line 146)
        build_quadratic_1d_216567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 146)
        build_quadratic_1d_call_result_216579 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), build_quadratic_1d_216567, *[J_216569, g_216571, s_216572], **kwargs_216578)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___216580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), build_quadratic_1d_call_result_216579, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_216581 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___216580, int_216566)
        
        # Assigning a type to the variable 'tuple_var_assignment_215642' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_215642', subscript_call_result_216581)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_216582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_216584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'self', False)
        # Obtaining the member 'J' of a type (line 146)
        J_216585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 37), self_216584, 'J')
        # Getting the type of 'self' (line 146)
        self_216586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 45), 'self', False)
        # Obtaining the member 'g' of a type (line 146)
        g_216587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 45), self_216586, 'g')
        # Getting the type of 's' (line 146)
        s_216588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 53), 's', False)
        # Processing the call keyword arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_216589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 61), 'self', False)
        # Obtaining the member 'diag' of a type (line 146)
        diag_216590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 61), self_216589, 'diag')
        keyword_216591 = diag_216590
        # Getting the type of 's0' (line 146)
        s0_216592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 75), 's0', False)
        keyword_216593 = s0_216592
        kwargs_216594 = {'diag': keyword_216591, 's0': keyword_216593}
        # Getting the type of 'build_quadratic_1d' (line 146)
        build_quadratic_1d_216583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 146)
        build_quadratic_1d_call_result_216595 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), build_quadratic_1d_216583, *[J_216585, g_216587, s_216588], **kwargs_216594)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___216596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), build_quadratic_1d_call_result_216595, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_216597 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___216596, int_216582)
        
        # Assigning a type to the variable 'tuple_var_assignment_215643' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_215643', subscript_call_result_216597)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_216598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to build_quadratic_1d(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_216600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'self', False)
        # Obtaining the member 'J' of a type (line 146)
        J_216601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 37), self_216600, 'J')
        # Getting the type of 'self' (line 146)
        self_216602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 45), 'self', False)
        # Obtaining the member 'g' of a type (line 146)
        g_216603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 45), self_216602, 'g')
        # Getting the type of 's' (line 146)
        s_216604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 53), 's', False)
        # Processing the call keyword arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_216605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 61), 'self', False)
        # Obtaining the member 'diag' of a type (line 146)
        diag_216606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 61), self_216605, 'diag')
        keyword_216607 = diag_216606
        # Getting the type of 's0' (line 146)
        s0_216608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 75), 's0', False)
        keyword_216609 = s0_216608
        kwargs_216610 = {'diag': keyword_216607, 's0': keyword_216609}
        # Getting the type of 'build_quadratic_1d' (line 146)
        build_quadratic_1d_216599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'build_quadratic_1d', False)
        # Calling build_quadratic_1d(args, kwargs) (line 146)
        build_quadratic_1d_call_result_216611 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), build_quadratic_1d_216599, *[J_216601, g_216603, s_216604], **kwargs_216610)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___216612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), build_quadratic_1d_call_result_216611, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_216613 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___216612, int_216598)
        
        # Assigning a type to the variable 'tuple_var_assignment_215644' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_215644', subscript_call_result_216613)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_215642' (line 146)
        tuple_var_assignment_215642_216614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_215642')
        # Assigning a type to the variable 'a' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'a', tuple_var_assignment_215642_216614)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_215643' (line 146)
        tuple_var_assignment_215643_216615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_215643')
        # Assigning a type to the variable 'b' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'b', tuple_var_assignment_215643_216615)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_215644' (line 146)
        tuple_var_assignment_215644_216616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_215644')
        # Assigning a type to the variable 'c' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 14), 'c', tuple_var_assignment_215644_216616)
        
        # Call to assert_equal(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'a' (line 147)
        a_216618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'a', False)
        float_216619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'float')
        # Processing the call keyword arguments (line 147)
        kwargs_216620 = {}
        # Getting the type of 'assert_equal' (line 147)
        assert_equal_216617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 147)
        assert_equal_call_result_216621 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assert_equal_216617, *[a_216618, float_216619], **kwargs_216620)
        
        
        # Call to assert_allclose(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'b' (line 148)
        b_216623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'b', False)
        float_216624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'float')
        # Processing the call keyword arguments (line 148)
        kwargs_216625 = {}
        # Getting the type of 'assert_allclose' (line 148)
        assert_allclose_216622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 148)
        assert_allclose_call_result_216626 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), assert_allclose_216622, *[b_216623, float_216624], **kwargs_216625)
        
        
        # Call to assert_allclose(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'c' (line 149)
        c_216628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'c', False)
        float_216629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 27), 'float')
        # Processing the call keyword arguments (line 149)
        kwargs_216630 = {}
        # Getting the type of 'assert_allclose' (line 149)
        assert_allclose_216627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 149)
        assert_allclose_call_result_216631 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assert_allclose_216627, *[c_216628, float_216629], **kwargs_216630)
        
        
        # ################# End of 'test_build_quadratic_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_build_quadratic_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_216632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_216632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_build_quadratic_1d'
        return stypy_return_type_216632


    @norecursion
    def test_minimize_quadratic_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_quadratic_1d'
        module_type_store = module_type_store.open_function_context('test_minimize_quadratic_1d', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_localization', localization)
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_function_name', 'TestQuadraticFunction.test_minimize_quadratic_1d')
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadraticFunction.test_minimize_quadratic_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadraticFunction.test_minimize_quadratic_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_quadratic_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_quadratic_1d(...)' code ##################

        
        # Assigning a Num to a Name (line 152):
        
        # Assigning a Num to a Name (line 152):
        int_216633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
        # Assigning a type to the variable 'a' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'a', int_216633)
        
        # Assigning a Num to a Name (line 153):
        
        # Assigning a Num to a Name (line 153):
        int_216634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 12), 'int')
        # Assigning a type to the variable 'b' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'b', int_216634)
        
        # Assigning a Call to a Tuple (line 155):
        
        # Assigning a Subscript to a Name (line 155):
        
        # Obtaining the type of the subscript
        int_216635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 8), 'int')
        
        # Call to minimize_quadratic_1d(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'a' (line 155)
        a_216637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 37), 'a', False)
        # Getting the type of 'b' (line 155)
        b_216638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 40), 'b', False)
        int_216639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 43), 'int')
        int_216640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 46), 'int')
        # Processing the call keyword arguments (line 155)
        kwargs_216641 = {}
        # Getting the type of 'minimize_quadratic_1d' (line 155)
        minimize_quadratic_1d_216636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'minimize_quadratic_1d', False)
        # Calling minimize_quadratic_1d(args, kwargs) (line 155)
        minimize_quadratic_1d_call_result_216642 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), minimize_quadratic_1d_216636, *[a_216637, b_216638, int_216639, int_216640], **kwargs_216641)
        
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___216643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), minimize_quadratic_1d_call_result_216642, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_216644 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), getitem___216643, int_216635)
        
        # Assigning a type to the variable 'tuple_var_assignment_215645' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'tuple_var_assignment_215645', subscript_call_result_216644)
        
        # Assigning a Subscript to a Name (line 155):
        
        # Obtaining the type of the subscript
        int_216645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 8), 'int')
        
        # Call to minimize_quadratic_1d(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'a' (line 155)
        a_216647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 37), 'a', False)
        # Getting the type of 'b' (line 155)
        b_216648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 40), 'b', False)
        int_216649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 43), 'int')
        int_216650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 46), 'int')
        # Processing the call keyword arguments (line 155)
        kwargs_216651 = {}
        # Getting the type of 'minimize_quadratic_1d' (line 155)
        minimize_quadratic_1d_216646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'minimize_quadratic_1d', False)
        # Calling minimize_quadratic_1d(args, kwargs) (line 155)
        minimize_quadratic_1d_call_result_216652 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), minimize_quadratic_1d_216646, *[a_216647, b_216648, int_216649, int_216650], **kwargs_216651)
        
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___216653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), minimize_quadratic_1d_call_result_216652, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_216654 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), getitem___216653, int_216645)
        
        # Assigning a type to the variable 'tuple_var_assignment_215646' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'tuple_var_assignment_215646', subscript_call_result_216654)
        
        # Assigning a Name to a Name (line 155):
        # Getting the type of 'tuple_var_assignment_215645' (line 155)
        tuple_var_assignment_215645_216655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'tuple_var_assignment_215645')
        # Assigning a type to the variable 't' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 't', tuple_var_assignment_215645_216655)
        
        # Assigning a Name to a Name (line 155):
        # Getting the type of 'tuple_var_assignment_215646' (line 155)
        tuple_var_assignment_215646_216656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'tuple_var_assignment_215646')
        # Assigning a type to the variable 'y' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'y', tuple_var_assignment_215646_216656)
        
        # Call to assert_equal(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 't' (line 156)
        t_216658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 't', False)
        int_216659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'int')
        # Processing the call keyword arguments (line 156)
        kwargs_216660 = {}
        # Getting the type of 'assert_equal' (line 156)
        assert_equal_216657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 156)
        assert_equal_call_result_216661 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), assert_equal_216657, *[t_216658, int_216659], **kwargs_216660)
        
        
        # Call to assert_equal(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'y' (line 157)
        y_216663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'y', False)
        # Getting the type of 'a' (line 157)
        a_216664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'a', False)
        # Getting the type of 't' (line 157)
        t_216665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 't', False)
        int_216666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 31), 'int')
        # Applying the binary operator '**' (line 157)
        result_pow_216667 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 28), '**', t_216665, int_216666)
        
        # Applying the binary operator '*' (line 157)
        result_mul_216668 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 24), '*', a_216664, result_pow_216667)
        
        # Getting the type of 'b' (line 157)
        b_216669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 35), 'b', False)
        # Getting the type of 't' (line 157)
        t_216670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 't', False)
        # Applying the binary operator '*' (line 157)
        result_mul_216671 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 35), '*', b_216669, t_216670)
        
        # Applying the binary operator '+' (line 157)
        result_add_216672 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 24), '+', result_mul_216668, result_mul_216671)
        
        # Processing the call keyword arguments (line 157)
        kwargs_216673 = {}
        # Getting the type of 'assert_equal' (line 157)
        assert_equal_216662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 157)
        assert_equal_call_result_216674 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), assert_equal_216662, *[y_216663, result_add_216672], **kwargs_216673)
        
        
        # Assigning a Call to a Tuple (line 159):
        
        # Assigning a Subscript to a Name (line 159):
        
        # Obtaining the type of the subscript
        int_216675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        
        # Call to minimize_quadratic_1d(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'a' (line 159)
        a_216677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'a', False)
        # Getting the type of 'b' (line 159)
        b_216678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 40), 'b', False)
        int_216679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'int')
        int_216680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 47), 'int')
        # Processing the call keyword arguments (line 159)
        kwargs_216681 = {}
        # Getting the type of 'minimize_quadratic_1d' (line 159)
        minimize_quadratic_1d_216676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'minimize_quadratic_1d', False)
        # Calling minimize_quadratic_1d(args, kwargs) (line 159)
        minimize_quadratic_1d_call_result_216682 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), minimize_quadratic_1d_216676, *[a_216677, b_216678, int_216679, int_216680], **kwargs_216681)
        
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___216683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), minimize_quadratic_1d_call_result_216682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_216684 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), getitem___216683, int_216675)
        
        # Assigning a type to the variable 'tuple_var_assignment_215647' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tuple_var_assignment_215647', subscript_call_result_216684)
        
        # Assigning a Subscript to a Name (line 159):
        
        # Obtaining the type of the subscript
        int_216685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        
        # Call to minimize_quadratic_1d(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'a' (line 159)
        a_216687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'a', False)
        # Getting the type of 'b' (line 159)
        b_216688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 40), 'b', False)
        int_216689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'int')
        int_216690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 47), 'int')
        # Processing the call keyword arguments (line 159)
        kwargs_216691 = {}
        # Getting the type of 'minimize_quadratic_1d' (line 159)
        minimize_quadratic_1d_216686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'minimize_quadratic_1d', False)
        # Calling minimize_quadratic_1d(args, kwargs) (line 159)
        minimize_quadratic_1d_call_result_216692 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), minimize_quadratic_1d_216686, *[a_216687, b_216688, int_216689, int_216690], **kwargs_216691)
        
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___216693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), minimize_quadratic_1d_call_result_216692, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_216694 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), getitem___216693, int_216685)
        
        # Assigning a type to the variable 'tuple_var_assignment_215648' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tuple_var_assignment_215648', subscript_call_result_216694)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'tuple_var_assignment_215647' (line 159)
        tuple_var_assignment_215647_216695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tuple_var_assignment_215647')
        # Assigning a type to the variable 't' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 't', tuple_var_assignment_215647_216695)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'tuple_var_assignment_215648' (line 159)
        tuple_var_assignment_215648_216696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tuple_var_assignment_215648')
        # Assigning a type to the variable 'y' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'y', tuple_var_assignment_215648_216696)
        
        # Call to assert_equal(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 't' (line 160)
        t_216698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 't', False)
        int_216699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'int')
        # Processing the call keyword arguments (line 160)
        kwargs_216700 = {}
        # Getting the type of 'assert_equal' (line 160)
        assert_equal_216697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 160)
        assert_equal_call_result_216701 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), assert_equal_216697, *[t_216698, int_216699], **kwargs_216700)
        
        
        # Call to assert_equal(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'y' (line 161)
        y_216703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'y', False)
        # Getting the type of 'a' (line 161)
        a_216704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'a', False)
        # Getting the type of 't' (line 161)
        t_216705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 't', False)
        int_216706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 31), 'int')
        # Applying the binary operator '**' (line 161)
        result_pow_216707 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 28), '**', t_216705, int_216706)
        
        # Applying the binary operator '*' (line 161)
        result_mul_216708 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 24), '*', a_216704, result_pow_216707)
        
        # Getting the type of 'b' (line 161)
        b_216709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'b', False)
        # Getting the type of 't' (line 161)
        t_216710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 't', False)
        # Applying the binary operator '*' (line 161)
        result_mul_216711 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 35), '*', b_216709, t_216710)
        
        # Applying the binary operator '+' (line 161)
        result_add_216712 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 24), '+', result_mul_216708, result_mul_216711)
        
        # Processing the call keyword arguments (line 161)
        kwargs_216713 = {}
        # Getting the type of 'assert_equal' (line 161)
        assert_equal_216702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 161)
        assert_equal_call_result_216714 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assert_equal_216702, *[y_216703, result_add_216712], **kwargs_216713)
        
        
        # Assigning a Call to a Tuple (line 163):
        
        # Assigning a Subscript to a Name (line 163):
        
        # Obtaining the type of the subscript
        int_216715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 8), 'int')
        
        # Call to minimize_quadratic_1d(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'a' (line 163)
        a_216717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'a', False)
        # Getting the type of 'b' (line 163)
        b_216718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 40), 'b', False)
        int_216719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 43), 'int')
        int_216720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 47), 'int')
        # Processing the call keyword arguments (line 163)
        kwargs_216721 = {}
        # Getting the type of 'minimize_quadratic_1d' (line 163)
        minimize_quadratic_1d_216716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'minimize_quadratic_1d', False)
        # Calling minimize_quadratic_1d(args, kwargs) (line 163)
        minimize_quadratic_1d_call_result_216722 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), minimize_quadratic_1d_216716, *[a_216717, b_216718, int_216719, int_216720], **kwargs_216721)
        
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___216723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), minimize_quadratic_1d_call_result_216722, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_216724 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), getitem___216723, int_216715)
        
        # Assigning a type to the variable 'tuple_var_assignment_215649' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'tuple_var_assignment_215649', subscript_call_result_216724)
        
        # Assigning a Subscript to a Name (line 163):
        
        # Obtaining the type of the subscript
        int_216725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 8), 'int')
        
        # Call to minimize_quadratic_1d(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'a' (line 163)
        a_216727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'a', False)
        # Getting the type of 'b' (line 163)
        b_216728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 40), 'b', False)
        int_216729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 43), 'int')
        int_216730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 47), 'int')
        # Processing the call keyword arguments (line 163)
        kwargs_216731 = {}
        # Getting the type of 'minimize_quadratic_1d' (line 163)
        minimize_quadratic_1d_216726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'minimize_quadratic_1d', False)
        # Calling minimize_quadratic_1d(args, kwargs) (line 163)
        minimize_quadratic_1d_call_result_216732 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), minimize_quadratic_1d_216726, *[a_216727, b_216728, int_216729, int_216730], **kwargs_216731)
        
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___216733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), minimize_quadratic_1d_call_result_216732, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_216734 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), getitem___216733, int_216725)
        
        # Assigning a type to the variable 'tuple_var_assignment_215650' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'tuple_var_assignment_215650', subscript_call_result_216734)
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'tuple_var_assignment_215649' (line 163)
        tuple_var_assignment_215649_216735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'tuple_var_assignment_215649')
        # Assigning a type to the variable 't' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 't', tuple_var_assignment_215649_216735)
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'tuple_var_assignment_215650' (line 163)
        tuple_var_assignment_215650_216736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'tuple_var_assignment_215650')
        # Assigning a type to the variable 'y' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'y', tuple_var_assignment_215650_216736)
        
        # Call to assert_equal(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 't' (line 164)
        t_216738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 't', False)
        float_216739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 24), 'float')
        # Processing the call keyword arguments (line 164)
        kwargs_216740 = {}
        # Getting the type of 'assert_equal' (line 164)
        assert_equal_216737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 164)
        assert_equal_call_result_216741 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), assert_equal_216737, *[t_216738, float_216739], **kwargs_216740)
        
        
        # Call to assert_equal(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'y' (line 165)
        y_216743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'y', False)
        # Getting the type of 'a' (line 165)
        a_216744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'a', False)
        # Getting the type of 't' (line 165)
        t_216745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 't', False)
        int_216746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 31), 'int')
        # Applying the binary operator '**' (line 165)
        result_pow_216747 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 28), '**', t_216745, int_216746)
        
        # Applying the binary operator '*' (line 165)
        result_mul_216748 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 24), '*', a_216744, result_pow_216747)
        
        # Getting the type of 'b' (line 165)
        b_216749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 35), 'b', False)
        # Getting the type of 't' (line 165)
        t_216750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 39), 't', False)
        # Applying the binary operator '*' (line 165)
        result_mul_216751 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 35), '*', b_216749, t_216750)
        
        # Applying the binary operator '+' (line 165)
        result_add_216752 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 24), '+', result_mul_216748, result_mul_216751)
        
        # Processing the call keyword arguments (line 165)
        kwargs_216753 = {}
        # Getting the type of 'assert_equal' (line 165)
        assert_equal_216742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 165)
        assert_equal_call_result_216754 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert_equal_216742, *[y_216743, result_add_216752], **kwargs_216753)
        
        
        # Assigning a Num to a Name (line 167):
        
        # Assigning a Num to a Name (line 167):
        int_216755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'int')
        # Assigning a type to the variable 'c' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'c', int_216755)
        
        # Assigning a Call to a Tuple (line 168):
        
        # Assigning a Subscript to a Name (line 168):
        
        # Obtaining the type of the subscript
        int_216756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'int')
        
        # Call to minimize_quadratic_1d(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'a' (line 168)
        a_216758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'a', False)
        # Getting the type of 'b' (line 168)
        b_216759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 40), 'b', False)
        int_216760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 43), 'int')
        int_216761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 47), 'int')
        # Processing the call keyword arguments (line 168)
        # Getting the type of 'c' (line 168)
        c_216762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 52), 'c', False)
        keyword_216763 = c_216762
        kwargs_216764 = {'c': keyword_216763}
        # Getting the type of 'minimize_quadratic_1d' (line 168)
        minimize_quadratic_1d_216757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'minimize_quadratic_1d', False)
        # Calling minimize_quadratic_1d(args, kwargs) (line 168)
        minimize_quadratic_1d_call_result_216765 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), minimize_quadratic_1d_216757, *[a_216758, b_216759, int_216760, int_216761], **kwargs_216764)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___216766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), minimize_quadratic_1d_call_result_216765, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_216767 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___216766, int_216756)
        
        # Assigning a type to the variable 'tuple_var_assignment_215651' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_215651', subscript_call_result_216767)
        
        # Assigning a Subscript to a Name (line 168):
        
        # Obtaining the type of the subscript
        int_216768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'int')
        
        # Call to minimize_quadratic_1d(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'a' (line 168)
        a_216770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'a', False)
        # Getting the type of 'b' (line 168)
        b_216771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 40), 'b', False)
        int_216772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 43), 'int')
        int_216773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 47), 'int')
        # Processing the call keyword arguments (line 168)
        # Getting the type of 'c' (line 168)
        c_216774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 52), 'c', False)
        keyword_216775 = c_216774
        kwargs_216776 = {'c': keyword_216775}
        # Getting the type of 'minimize_quadratic_1d' (line 168)
        minimize_quadratic_1d_216769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'minimize_quadratic_1d', False)
        # Calling minimize_quadratic_1d(args, kwargs) (line 168)
        minimize_quadratic_1d_call_result_216777 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), minimize_quadratic_1d_216769, *[a_216770, b_216771, int_216772, int_216773], **kwargs_216776)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___216778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), minimize_quadratic_1d_call_result_216777, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_216779 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___216778, int_216768)
        
        # Assigning a type to the variable 'tuple_var_assignment_215652' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_215652', subscript_call_result_216779)
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'tuple_var_assignment_215651' (line 168)
        tuple_var_assignment_215651_216780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_215651')
        # Assigning a type to the variable 't' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 't', tuple_var_assignment_215651_216780)
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'tuple_var_assignment_215652' (line 168)
        tuple_var_assignment_215652_216781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_215652')
        # Assigning a type to the variable 'y' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'y', tuple_var_assignment_215652_216781)
        
        # Call to assert_equal(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 't' (line 169)
        t_216783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 't', False)
        float_216784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'float')
        # Processing the call keyword arguments (line 169)
        kwargs_216785 = {}
        # Getting the type of 'assert_equal' (line 169)
        assert_equal_216782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 169)
        assert_equal_call_result_216786 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), assert_equal_216782, *[t_216783, float_216784], **kwargs_216785)
        
        
        # Call to assert_equal(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'y' (line 170)
        y_216788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'y', False)
        # Getting the type of 'a' (line 170)
        a_216789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'a', False)
        # Getting the type of 't' (line 170)
        t_216790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 't', False)
        int_216791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'int')
        # Applying the binary operator '**' (line 170)
        result_pow_216792 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 28), '**', t_216790, int_216791)
        
        # Applying the binary operator '*' (line 170)
        result_mul_216793 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 24), '*', a_216789, result_pow_216792)
        
        # Getting the type of 'b' (line 170)
        b_216794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 35), 'b', False)
        # Getting the type of 't' (line 170)
        t_216795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 39), 't', False)
        # Applying the binary operator '*' (line 170)
        result_mul_216796 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 35), '*', b_216794, t_216795)
        
        # Applying the binary operator '+' (line 170)
        result_add_216797 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 24), '+', result_mul_216793, result_mul_216796)
        
        # Getting the type of 'c' (line 170)
        c_216798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 43), 'c', False)
        # Applying the binary operator '+' (line 170)
        result_add_216799 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 41), '+', result_add_216797, c_216798)
        
        # Processing the call keyword arguments (line 170)
        kwargs_216800 = {}
        # Getting the type of 'assert_equal' (line 170)
        assert_equal_216787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 170)
        assert_equal_call_result_216801 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), assert_equal_216787, *[y_216788, result_add_216799], **kwargs_216800)
        
        
        # ################# End of 'test_minimize_quadratic_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_quadratic_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_216802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_216802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_quadratic_1d'
        return stypy_return_type_216802


    @norecursion
    def test_evaluate_quadratic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_evaluate_quadratic'
        module_type_store = module_type_store.open_function_context('test_evaluate_quadratic', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_localization', localization)
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_function_name', 'TestQuadraticFunction.test_evaluate_quadratic')
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadraticFunction.test_evaluate_quadratic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadraticFunction.test_evaluate_quadratic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_evaluate_quadratic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_evaluate_quadratic(...)' code ##################

        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to array(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Obtaining an instance of the builtin type 'list' (line 173)
        list_216805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 173)
        # Adding element type (line 173)
        float_216806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 21), list_216805, float_216806)
        # Adding element type (line 173)
        float_216807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 21), list_216805, float_216807)
        
        # Processing the call keyword arguments (line 173)
        kwargs_216808 = {}
        # Getting the type of 'np' (line 173)
        np_216803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 173)
        array_216804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 12), np_216803, 'array')
        # Calling array(args, kwargs) (line 173)
        array_call_result_216809 = invoke(stypy.reporting.localization.Localization(__file__, 173, 12), array_216804, *[list_216805], **kwargs_216808)
        
        # Assigning a type to the variable 's' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 's', array_call_result_216809)
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to evaluate_quadratic(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'self' (line 175)
        self_216811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'self', False)
        # Obtaining the member 'J' of a type (line 175)
        J_216812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 35), self_216811, 'J')
        # Getting the type of 'self' (line 175)
        self_216813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'self', False)
        # Obtaining the member 'g' of a type (line 175)
        g_216814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 43), self_216813, 'g')
        # Getting the type of 's' (line 175)
        s_216815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 51), 's', False)
        # Processing the call keyword arguments (line 175)
        kwargs_216816 = {}
        # Getting the type of 'evaluate_quadratic' (line 175)
        evaluate_quadratic_216810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'evaluate_quadratic', False)
        # Calling evaluate_quadratic(args, kwargs) (line 175)
        evaluate_quadratic_call_result_216817 = invoke(stypy.reporting.localization.Localization(__file__, 175, 16), evaluate_quadratic_216810, *[J_216812, g_216814, s_216815], **kwargs_216816)
        
        # Assigning a type to the variable 'value' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'value', evaluate_quadratic_call_result_216817)
        
        # Call to assert_equal(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'value' (line 176)
        value_216819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), 'value', False)
        float_216820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'float')
        # Processing the call keyword arguments (line 176)
        kwargs_216821 = {}
        # Getting the type of 'assert_equal' (line 176)
        assert_equal_216818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 176)
        assert_equal_call_result_216822 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assert_equal_216818, *[value_216819, float_216820], **kwargs_216821)
        
        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to evaluate_quadratic(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'self' (line 178)
        self_216824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'self', False)
        # Obtaining the member 'J' of a type (line 178)
        J_216825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 35), self_216824, 'J')
        # Getting the type of 'self' (line 178)
        self_216826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 43), 'self', False)
        # Obtaining the member 'g' of a type (line 178)
        g_216827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 43), self_216826, 'g')
        # Getting the type of 's' (line 178)
        s_216828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 51), 's', False)
        # Processing the call keyword arguments (line 178)
        # Getting the type of 'self' (line 178)
        self_216829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 59), 'self', False)
        # Obtaining the member 'diag' of a type (line 178)
        diag_216830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 59), self_216829, 'diag')
        keyword_216831 = diag_216830
        kwargs_216832 = {'diag': keyword_216831}
        # Getting the type of 'evaluate_quadratic' (line 178)
        evaluate_quadratic_216823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'evaluate_quadratic', False)
        # Calling evaluate_quadratic(args, kwargs) (line 178)
        evaluate_quadratic_call_result_216833 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), evaluate_quadratic_216823, *[J_216825, g_216827, s_216828], **kwargs_216832)
        
        # Assigning a type to the variable 'value' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'value', evaluate_quadratic_call_result_216833)
        
        # Call to assert_equal(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'value' (line 179)
        value_216835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'value', False)
        float_216836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 28), 'float')
        # Processing the call keyword arguments (line 179)
        kwargs_216837 = {}
        # Getting the type of 'assert_equal' (line 179)
        assert_equal_216834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 179)
        assert_equal_call_result_216838 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), assert_equal_216834, *[value_216835, float_216836], **kwargs_216837)
        
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to array(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_216841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        # Adding element type (line 181)
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_216842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        # Adding element type (line 181)
        float_216843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 22), list_216842, float_216843)
        # Adding element type (line 181)
        float_216844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 22), list_216842, float_216844)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 21), list_216841, list_216842)
        # Adding element type (line 181)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_216845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        float_216846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 21), list_216845, float_216846)
        # Adding element type (line 182)
        float_216847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 21), list_216845, float_216847)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 21), list_216841, list_216845)
        # Adding element type (line 181)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_216848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        float_216849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_216848, float_216849)
        # Adding element type (line 183)
        float_216850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_216848, float_216850)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 21), list_216841, list_216848)
        
        # Processing the call keyword arguments (line 181)
        kwargs_216851 = {}
        # Getting the type of 'np' (line 181)
        np_216839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 181)
        array_216840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), np_216839, 'array')
        # Calling array(args, kwargs) (line 181)
        array_call_result_216852 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), array_216840, *[list_216841], **kwargs_216851)
        
        # Assigning a type to the variable 's' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 's', array_call_result_216852)
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to evaluate_quadratic(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'self' (line 185)
        self_216854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 36), 'self', False)
        # Obtaining the member 'J' of a type (line 185)
        J_216855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 36), self_216854, 'J')
        # Getting the type of 'self' (line 185)
        self_216856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 44), 'self', False)
        # Obtaining the member 'g' of a type (line 185)
        g_216857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 44), self_216856, 'g')
        # Getting the type of 's' (line 185)
        s_216858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 52), 's', False)
        # Processing the call keyword arguments (line 185)
        kwargs_216859 = {}
        # Getting the type of 'evaluate_quadratic' (line 185)
        evaluate_quadratic_216853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'evaluate_quadratic', False)
        # Calling evaluate_quadratic(args, kwargs) (line 185)
        evaluate_quadratic_call_result_216860 = invoke(stypy.reporting.localization.Localization(__file__, 185, 17), evaluate_quadratic_216853, *[J_216855, g_216857, s_216858], **kwargs_216859)
        
        # Assigning a type to the variable 'values' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'values', evaluate_quadratic_call_result_216860)
        
        # Call to assert_allclose(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'values' (line 186)
        values_216862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'values', False)
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_216863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        # Adding element type (line 186)
        float_216864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 32), list_216863, float_216864)
        # Adding element type (line 186)
        float_216865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 32), list_216863, float_216865)
        # Adding element type (line 186)
        float_216866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 32), list_216863, float_216866)
        
        # Processing the call keyword arguments (line 186)
        kwargs_216867 = {}
        # Getting the type of 'assert_allclose' (line 186)
        assert_allclose_216861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 186)
        assert_allclose_call_result_216868 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), assert_allclose_216861, *[values_216862, list_216863], **kwargs_216867)
        
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to evaluate_quadratic(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_216870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 36), 'self', False)
        # Obtaining the member 'J' of a type (line 188)
        J_216871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 36), self_216870, 'J')
        # Getting the type of 'self' (line 188)
        self_216872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 44), 'self', False)
        # Obtaining the member 'g' of a type (line 188)
        g_216873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 44), self_216872, 'g')
        # Getting the type of 's' (line 188)
        s_216874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 52), 's', False)
        # Processing the call keyword arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_216875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 60), 'self', False)
        # Obtaining the member 'diag' of a type (line 188)
        diag_216876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 60), self_216875, 'diag')
        keyword_216877 = diag_216876
        kwargs_216878 = {'diag': keyword_216877}
        # Getting the type of 'evaluate_quadratic' (line 188)
        evaluate_quadratic_216869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 'evaluate_quadratic', False)
        # Calling evaluate_quadratic(args, kwargs) (line 188)
        evaluate_quadratic_call_result_216879 = invoke(stypy.reporting.localization.Localization(__file__, 188, 17), evaluate_quadratic_216869, *[J_216871, g_216873, s_216874], **kwargs_216878)
        
        # Assigning a type to the variable 'values' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'values', evaluate_quadratic_call_result_216879)
        
        # Call to assert_allclose(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'values' (line 189)
        values_216881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'values', False)
        
        # Obtaining an instance of the builtin type 'list' (line 189)
        list_216882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 189)
        # Adding element type (line 189)
        float_216883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 32), list_216882, float_216883)
        # Adding element type (line 189)
        float_216884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 32), list_216882, float_216884)
        # Adding element type (line 189)
        float_216885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 32), list_216882, float_216885)
        
        # Processing the call keyword arguments (line 189)
        kwargs_216886 = {}
        # Getting the type of 'assert_allclose' (line 189)
        assert_allclose_216880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 189)
        assert_allclose_call_result_216887 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assert_allclose_216880, *[values_216881, list_216882], **kwargs_216886)
        
        
        # ################# End of 'test_evaluate_quadratic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_evaluate_quadratic' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_216888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_216888)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_evaluate_quadratic'
        return stypy_return_type_216888


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 117, 0, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadraticFunction.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestQuadraticFunction' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'TestQuadraticFunction', TestQuadraticFunction)
# Declaration of the 'TestTrustRegion' class

class TestTrustRegion(object, ):

    @norecursion
    def test_intersect(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_intersect'
        module_type_store = module_type_store.open_function_context('test_intersect', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_localization', localization)
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_function_name', 'TestTrustRegion.test_intersect')
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_param_names_list', [])
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTrustRegion.test_intersect.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTrustRegion.test_intersect', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_intersect', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_intersect(...)' code ##################

        
        # Assigning a Num to a Name (line 194):
        
        # Assigning a Num to a Name (line 194):
        float_216889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 16), 'float')
        # Assigning a type to the variable 'Delta' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'Delta', float_216889)
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to zeros(...): (line 196)
        # Processing the call arguments (line 196)
        int_216892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 21), 'int')
        # Processing the call keyword arguments (line 196)
        kwargs_216893 = {}
        # Getting the type of 'np' (line 196)
        np_216890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 196)
        zeros_216891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), np_216890, 'zeros')
        # Calling zeros(args, kwargs) (line 196)
        zeros_call_result_216894 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), zeros_216891, *[int_216892], **kwargs_216893)
        
        # Assigning a type to the variable 'x' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'x', zeros_call_result_216894)
        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to array(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_216897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        float_216898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_216897, float_216898)
        # Adding element type (line 197)
        float_216899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_216897, float_216899)
        # Adding element type (line 197)
        float_216900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_216897, float_216900)
        
        # Processing the call keyword arguments (line 197)
        kwargs_216901 = {}
        # Getting the type of 'np' (line 197)
        np_216895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 197)
        array_216896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), np_216895, 'array')
        # Calling array(args, kwargs) (line 197)
        array_call_result_216902 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), array_216896, *[list_216897], **kwargs_216901)
        
        # Assigning a type to the variable 's' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 's', array_call_result_216902)
        
        # Assigning a Call to a Tuple (line 198):
        
        # Assigning a Subscript to a Name (line 198):
        
        # Obtaining the type of the subscript
        int_216903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 8), 'int')
        
        # Call to intersect_trust_region(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'x' (line 198)
        x_216905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 46), 'x', False)
        # Getting the type of 's' (line 198)
        s_216906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 49), 's', False)
        # Getting the type of 'Delta' (line 198)
        Delta_216907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 52), 'Delta', False)
        # Processing the call keyword arguments (line 198)
        kwargs_216908 = {}
        # Getting the type of 'intersect_trust_region' (line 198)
        intersect_trust_region_216904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'intersect_trust_region', False)
        # Calling intersect_trust_region(args, kwargs) (line 198)
        intersect_trust_region_call_result_216909 = invoke(stypy.reporting.localization.Localization(__file__, 198, 23), intersect_trust_region_216904, *[x_216905, s_216906, Delta_216907], **kwargs_216908)
        
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___216910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), intersect_trust_region_call_result_216909, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_216911 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), getitem___216910, int_216903)
        
        # Assigning a type to the variable 'tuple_var_assignment_215653' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_215653', subscript_call_result_216911)
        
        # Assigning a Subscript to a Name (line 198):
        
        # Obtaining the type of the subscript
        int_216912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 8), 'int')
        
        # Call to intersect_trust_region(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'x' (line 198)
        x_216914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 46), 'x', False)
        # Getting the type of 's' (line 198)
        s_216915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 49), 's', False)
        # Getting the type of 'Delta' (line 198)
        Delta_216916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 52), 'Delta', False)
        # Processing the call keyword arguments (line 198)
        kwargs_216917 = {}
        # Getting the type of 'intersect_trust_region' (line 198)
        intersect_trust_region_216913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'intersect_trust_region', False)
        # Calling intersect_trust_region(args, kwargs) (line 198)
        intersect_trust_region_call_result_216918 = invoke(stypy.reporting.localization.Localization(__file__, 198, 23), intersect_trust_region_216913, *[x_216914, s_216915, Delta_216916], **kwargs_216917)
        
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___216919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), intersect_trust_region_call_result_216918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_216920 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), getitem___216919, int_216912)
        
        # Assigning a type to the variable 'tuple_var_assignment_215654' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_215654', subscript_call_result_216920)
        
        # Assigning a Name to a Name (line 198):
        # Getting the type of 'tuple_var_assignment_215653' (line 198)
        tuple_var_assignment_215653_216921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_215653')
        # Assigning a type to the variable 't_neg' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 't_neg', tuple_var_assignment_215653_216921)
        
        # Assigning a Name to a Name (line 198):
        # Getting the type of 'tuple_var_assignment_215654' (line 198)
        tuple_var_assignment_215654_216922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_215654')
        # Assigning a type to the variable 't_pos' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 't_pos', tuple_var_assignment_215654_216922)
        
        # Call to assert_equal(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 't_neg' (line 199)
        t_neg_216924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 't_neg', False)
        int_216925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 28), 'int')
        # Processing the call keyword arguments (line 199)
        kwargs_216926 = {}
        # Getting the type of 'assert_equal' (line 199)
        assert_equal_216923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 199)
        assert_equal_call_result_216927 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), assert_equal_216923, *[t_neg_216924, int_216925], **kwargs_216926)
        
        
        # Call to assert_equal(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 't_pos' (line 200)
        t_pos_216929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 't_pos', False)
        int_216930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 28), 'int')
        # Processing the call keyword arguments (line 200)
        kwargs_216931 = {}
        # Getting the type of 'assert_equal' (line 200)
        assert_equal_216928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 200)
        assert_equal_call_result_216932 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), assert_equal_216928, *[t_pos_216929, int_216930], **kwargs_216931)
        
        
        # Assigning a Call to a Name (line 202):
        
        # Assigning a Call to a Name (line 202):
        
        # Call to array(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_216935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        float_216936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 21), list_216935, float_216936)
        # Adding element type (line 202)
        float_216937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 21), list_216935, float_216937)
        # Adding element type (line 202)
        float_216938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 21), list_216935, float_216938)
        
        # Processing the call keyword arguments (line 202)
        kwargs_216939 = {}
        # Getting the type of 'np' (line 202)
        np_216933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 202)
        array_216934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), np_216933, 'array')
        # Calling array(args, kwargs) (line 202)
        array_call_result_216940 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), array_216934, *[list_216935], **kwargs_216939)
        
        # Assigning a type to the variable 's' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 's', array_call_result_216940)
        
        # Assigning a Call to a Tuple (line 203):
        
        # Assigning a Subscript to a Name (line 203):
        
        # Obtaining the type of the subscript
        int_216941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 8), 'int')
        
        # Call to intersect_trust_region(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'x' (line 203)
        x_216943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 46), 'x', False)
        # Getting the type of 's' (line 203)
        s_216944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 49), 's', False)
        # Getting the type of 'Delta' (line 203)
        Delta_216945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 52), 'Delta', False)
        # Processing the call keyword arguments (line 203)
        kwargs_216946 = {}
        # Getting the type of 'intersect_trust_region' (line 203)
        intersect_trust_region_216942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'intersect_trust_region', False)
        # Calling intersect_trust_region(args, kwargs) (line 203)
        intersect_trust_region_call_result_216947 = invoke(stypy.reporting.localization.Localization(__file__, 203, 23), intersect_trust_region_216942, *[x_216943, s_216944, Delta_216945], **kwargs_216946)
        
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___216948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), intersect_trust_region_call_result_216947, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_216949 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), getitem___216948, int_216941)
        
        # Assigning a type to the variable 'tuple_var_assignment_215655' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tuple_var_assignment_215655', subscript_call_result_216949)
        
        # Assigning a Subscript to a Name (line 203):
        
        # Obtaining the type of the subscript
        int_216950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 8), 'int')
        
        # Call to intersect_trust_region(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'x' (line 203)
        x_216952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 46), 'x', False)
        # Getting the type of 's' (line 203)
        s_216953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 49), 's', False)
        # Getting the type of 'Delta' (line 203)
        Delta_216954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 52), 'Delta', False)
        # Processing the call keyword arguments (line 203)
        kwargs_216955 = {}
        # Getting the type of 'intersect_trust_region' (line 203)
        intersect_trust_region_216951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'intersect_trust_region', False)
        # Calling intersect_trust_region(args, kwargs) (line 203)
        intersect_trust_region_call_result_216956 = invoke(stypy.reporting.localization.Localization(__file__, 203, 23), intersect_trust_region_216951, *[x_216952, s_216953, Delta_216954], **kwargs_216955)
        
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___216957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), intersect_trust_region_call_result_216956, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_216958 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), getitem___216957, int_216950)
        
        # Assigning a type to the variable 'tuple_var_assignment_215656' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tuple_var_assignment_215656', subscript_call_result_216958)
        
        # Assigning a Name to a Name (line 203):
        # Getting the type of 'tuple_var_assignment_215655' (line 203)
        tuple_var_assignment_215655_216959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tuple_var_assignment_215655')
        # Assigning a type to the variable 't_neg' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 't_neg', tuple_var_assignment_215655_216959)
        
        # Assigning a Name to a Name (line 203):
        # Getting the type of 'tuple_var_assignment_215656' (line 203)
        tuple_var_assignment_215656_216960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tuple_var_assignment_215656')
        # Assigning a type to the variable 't_pos' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 't_pos', tuple_var_assignment_215656_216960)
        
        # Call to assert_allclose(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 't_neg' (line 204)
        t_neg_216962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 't_neg', False)
        
        int_216963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 32), 'int')
        float_216964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 35), 'float')
        # Applying the binary operator '**' (line 204)
        result_pow_216965 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 32), '**', int_216963, float_216964)
        
        # Applying the 'usub' unary operator (line 204)
        result___neg___216966 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 31), 'usub', result_pow_216965)
        
        # Processing the call keyword arguments (line 204)
        kwargs_216967 = {}
        # Getting the type of 'assert_allclose' (line 204)
        assert_allclose_216961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 204)
        assert_allclose_call_result_216968 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), assert_allclose_216961, *[t_neg_216962, result___neg___216966], **kwargs_216967)
        
        
        # Call to assert_allclose(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 't_pos' (line 205)
        t_pos_216970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 't_pos', False)
        int_216971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 31), 'int')
        float_216972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 34), 'float')
        # Applying the binary operator '**' (line 205)
        result_pow_216973 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 31), '**', int_216971, float_216972)
        
        # Processing the call keyword arguments (line 205)
        kwargs_216974 = {}
        # Getting the type of 'assert_allclose' (line 205)
        assert_allclose_216969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 205)
        assert_allclose_call_result_216975 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), assert_allclose_216969, *[t_pos_216970, result_pow_216973], **kwargs_216974)
        
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to array(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_216978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        float_216979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 21), list_216978, float_216979)
        # Adding element type (line 207)
        float_216980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 21), list_216978, float_216980)
        # Adding element type (line 207)
        int_216981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 21), list_216978, int_216981)
        
        # Processing the call keyword arguments (line 207)
        kwargs_216982 = {}
        # Getting the type of 'np' (line 207)
        np_216976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 207)
        array_216977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), np_216976, 'array')
        # Calling array(args, kwargs) (line 207)
        array_call_result_216983 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), array_216977, *[list_216978], **kwargs_216982)
        
        # Assigning a type to the variable 'x' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'x', array_call_result_216983)
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to array(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_216986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        int_216987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 21), list_216986, int_216987)
        # Adding element type (line 208)
        int_216988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 21), list_216986, int_216988)
        # Adding element type (line 208)
        float_216989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 21), list_216986, float_216989)
        
        # Processing the call keyword arguments (line 208)
        kwargs_216990 = {}
        # Getting the type of 'np' (line 208)
        np_216984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 208)
        array_216985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), np_216984, 'array')
        # Calling array(args, kwargs) (line 208)
        array_call_result_216991 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), array_216985, *[list_216986], **kwargs_216990)
        
        # Assigning a type to the variable 's' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 's', array_call_result_216991)
        
        # Assigning a Call to a Tuple (line 209):
        
        # Assigning a Subscript to a Name (line 209):
        
        # Obtaining the type of the subscript
        int_216992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 8), 'int')
        
        # Call to intersect_trust_region(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'x' (line 209)
        x_216994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 46), 'x', False)
        # Getting the type of 's' (line 209)
        s_216995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 49), 's', False)
        # Getting the type of 'Delta' (line 209)
        Delta_216996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 52), 'Delta', False)
        # Processing the call keyword arguments (line 209)
        kwargs_216997 = {}
        # Getting the type of 'intersect_trust_region' (line 209)
        intersect_trust_region_216993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 23), 'intersect_trust_region', False)
        # Calling intersect_trust_region(args, kwargs) (line 209)
        intersect_trust_region_call_result_216998 = invoke(stypy.reporting.localization.Localization(__file__, 209, 23), intersect_trust_region_216993, *[x_216994, s_216995, Delta_216996], **kwargs_216997)
        
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___216999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), intersect_trust_region_call_result_216998, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_217000 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), getitem___216999, int_216992)
        
        # Assigning a type to the variable 'tuple_var_assignment_215657' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tuple_var_assignment_215657', subscript_call_result_217000)
        
        # Assigning a Subscript to a Name (line 209):
        
        # Obtaining the type of the subscript
        int_217001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 8), 'int')
        
        # Call to intersect_trust_region(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'x' (line 209)
        x_217003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 46), 'x', False)
        # Getting the type of 's' (line 209)
        s_217004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 49), 's', False)
        # Getting the type of 'Delta' (line 209)
        Delta_217005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 52), 'Delta', False)
        # Processing the call keyword arguments (line 209)
        kwargs_217006 = {}
        # Getting the type of 'intersect_trust_region' (line 209)
        intersect_trust_region_217002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 23), 'intersect_trust_region', False)
        # Calling intersect_trust_region(args, kwargs) (line 209)
        intersect_trust_region_call_result_217007 = invoke(stypy.reporting.localization.Localization(__file__, 209, 23), intersect_trust_region_217002, *[x_217003, s_217004, Delta_217005], **kwargs_217006)
        
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___217008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), intersect_trust_region_call_result_217007, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_217009 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), getitem___217008, int_217001)
        
        # Assigning a type to the variable 'tuple_var_assignment_215658' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tuple_var_assignment_215658', subscript_call_result_217009)
        
        # Assigning a Name to a Name (line 209):
        # Getting the type of 'tuple_var_assignment_215657' (line 209)
        tuple_var_assignment_215657_217010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tuple_var_assignment_215657')
        # Assigning a type to the variable 't_neg' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 't_neg', tuple_var_assignment_215657_217010)
        
        # Assigning a Name to a Name (line 209):
        # Getting the type of 'tuple_var_assignment_215658' (line 209)
        tuple_var_assignment_215658_217011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tuple_var_assignment_215658')
        # Assigning a type to the variable 't_pos' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 't_pos', tuple_var_assignment_215658_217011)
        
        # Call to assert_allclose(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 't_neg' (line 210)
        t_neg_217013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), 't_neg', False)
        
        int_217014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 32), 'int')
        float_217015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 35), 'float')
        # Applying the binary operator '**' (line 210)
        result_pow_217016 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 32), '**', int_217014, float_217015)
        
        # Applying the 'usub' unary operator (line 210)
        result___neg___217017 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 31), 'usub', result_pow_217016)
        
        # Processing the call keyword arguments (line 210)
        kwargs_217018 = {}
        # Getting the type of 'assert_allclose' (line 210)
        assert_allclose_217012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 210)
        assert_allclose_call_result_217019 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), assert_allclose_217012, *[t_neg_217013, result___neg___217017], **kwargs_217018)
        
        
        # Call to assert_allclose(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 't_pos' (line 211)
        t_pos_217021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 't_pos', False)
        int_217022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 31), 'int')
        float_217023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 34), 'float')
        # Applying the binary operator '**' (line 211)
        result_pow_217024 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 31), '**', int_217022, float_217023)
        
        # Processing the call keyword arguments (line 211)
        kwargs_217025 = {}
        # Getting the type of 'assert_allclose' (line 211)
        assert_allclose_217020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 211)
        assert_allclose_call_result_217026 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assert_allclose_217020, *[t_pos_217021, result_pow_217024], **kwargs_217025)
        
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to ones(...): (line 213)
        # Processing the call arguments (line 213)
        int_217029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 20), 'int')
        # Processing the call keyword arguments (line 213)
        kwargs_217030 = {}
        # Getting the type of 'np' (line 213)
        np_217027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 213)
        ones_217028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), np_217027, 'ones')
        # Calling ones(args, kwargs) (line 213)
        ones_call_result_217031 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), ones_217028, *[int_217029], **kwargs_217030)
        
        # Assigning a type to the variable 'x' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'x', ones_call_result_217031)
        
        # Call to assert_raises(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'ValueError' (line 214)
        ValueError_217033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 22), 'ValueError', False)
        # Getting the type of 'intersect_trust_region' (line 214)
        intersect_trust_region_217034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'intersect_trust_region', False)
        # Getting the type of 'x' (line 214)
        x_217035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 58), 'x', False)
        # Getting the type of 's' (line 214)
        s_217036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 61), 's', False)
        # Getting the type of 'Delta' (line 214)
        Delta_217037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 64), 'Delta', False)
        # Processing the call keyword arguments (line 214)
        kwargs_217038 = {}
        # Getting the type of 'assert_raises' (line 214)
        assert_raises_217032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 214)
        assert_raises_call_result_217039 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assert_raises_217032, *[ValueError_217033, intersect_trust_region_217034, x_217035, s_217036, Delta_217037], **kwargs_217038)
        
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to zeros(...): (line 216)
        # Processing the call arguments (line 216)
        int_217042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 21), 'int')
        # Processing the call keyword arguments (line 216)
        kwargs_217043 = {}
        # Getting the type of 'np' (line 216)
        np_217040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 216)
        zeros_217041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), np_217040, 'zeros')
        # Calling zeros(args, kwargs) (line 216)
        zeros_call_result_217044 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), zeros_217041, *[int_217042], **kwargs_217043)
        
        # Assigning a type to the variable 'x' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'x', zeros_call_result_217044)
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to zeros(...): (line 217)
        # Processing the call arguments (line 217)
        int_217047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 21), 'int')
        # Processing the call keyword arguments (line 217)
        kwargs_217048 = {}
        # Getting the type of 'np' (line 217)
        np_217045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 217)
        zeros_217046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), np_217045, 'zeros')
        # Calling zeros(args, kwargs) (line 217)
        zeros_call_result_217049 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), zeros_217046, *[int_217047], **kwargs_217048)
        
        # Assigning a type to the variable 's' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 's', zeros_call_result_217049)
        
        # Call to assert_raises(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'ValueError' (line 218)
        ValueError_217051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'ValueError', False)
        # Getting the type of 'intersect_trust_region' (line 218)
        intersect_trust_region_217052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'intersect_trust_region', False)
        # Getting the type of 'x' (line 218)
        x_217053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 58), 'x', False)
        # Getting the type of 's' (line 218)
        s_217054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 61), 's', False)
        # Getting the type of 'Delta' (line 218)
        Delta_217055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 64), 'Delta', False)
        # Processing the call keyword arguments (line 218)
        kwargs_217056 = {}
        # Getting the type of 'assert_raises' (line 218)
        assert_raises_217050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 218)
        assert_raises_call_result_217057 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), assert_raises_217050, *[ValueError_217051, intersect_trust_region_217052, x_217053, s_217054, Delta_217055], **kwargs_217056)
        
        
        # ################# End of 'test_intersect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_intersect' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_217058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217058)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_intersect'
        return stypy_return_type_217058


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTrustRegion.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTrustRegion' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'TestTrustRegion', TestTrustRegion)

@norecursion
def test_reflective_transformation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_reflective_transformation'
    module_type_store = module_type_store.open_function_context('test_reflective_transformation', 221, 0, False)
    
    # Passed parameters checking function
    test_reflective_transformation.stypy_localization = localization
    test_reflective_transformation.stypy_type_of_self = None
    test_reflective_transformation.stypy_type_store = module_type_store
    test_reflective_transformation.stypy_function_name = 'test_reflective_transformation'
    test_reflective_transformation.stypy_param_names_list = []
    test_reflective_transformation.stypy_varargs_param_name = None
    test_reflective_transformation.stypy_kwargs_param_name = None
    test_reflective_transformation.stypy_call_defaults = defaults
    test_reflective_transformation.stypy_call_varargs = varargs
    test_reflective_transformation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_reflective_transformation', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_reflective_transformation', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_reflective_transformation(...)' code ##################

    
    # Assigning a Call to a Name (line 222):
    
    # Assigning a Call to a Name (line 222):
    
    # Call to array(...): (line 222)
    # Processing the call arguments (line 222)
    
    # Obtaining an instance of the builtin type 'list' (line 222)
    list_217061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 222)
    # Adding element type (line 222)
    int_217062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 18), list_217061, int_217062)
    # Adding element type (line 222)
    int_217063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 18), list_217061, int_217063)
    
    # Processing the call keyword arguments (line 222)
    # Getting the type of 'float' (line 222)
    float_217064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 34), 'float', False)
    keyword_217065 = float_217064
    kwargs_217066 = {'dtype': keyword_217065}
    # Getting the type of 'np' (line 222)
    np_217059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 222)
    array_217060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 9), np_217059, 'array')
    # Calling array(args, kwargs) (line 222)
    array_call_result_217067 = invoke(stypy.reporting.localization.Localization(__file__, 222, 9), array_217060, *[list_217061], **kwargs_217066)
    
    # Assigning a type to the variable 'lb' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'lb', array_call_result_217067)
    
    # Assigning a Call to a Name (line 223):
    
    # Assigning a Call to a Name (line 223):
    
    # Call to array(...): (line 223)
    # Processing the call arguments (line 223)
    
    # Obtaining an instance of the builtin type 'list' (line 223)
    list_217070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 223)
    # Adding element type (line 223)
    int_217071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 18), list_217070, int_217071)
    # Adding element type (line 223)
    int_217072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 18), list_217070, int_217072)
    
    # Processing the call keyword arguments (line 223)
    # Getting the type of 'float' (line 223)
    float_217073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'float', False)
    keyword_217074 = float_217073
    kwargs_217075 = {'dtype': keyword_217074}
    # Getting the type of 'np' (line 223)
    np_217068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 223)
    array_217069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 9), np_217068, 'array')
    # Calling array(args, kwargs) (line 223)
    array_call_result_217076 = invoke(stypy.reporting.localization.Localization(__file__, 223, 9), array_217069, *[list_217070], **kwargs_217075)
    
    # Assigning a type to the variable 'ub' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'ub', array_call_result_217076)
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to array(...): (line 225)
    # Processing the call arguments (line 225)
    
    # Obtaining an instance of the builtin type 'list' (line 225)
    list_217079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 225)
    # Adding element type (line 225)
    int_217080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 17), list_217079, int_217080)
    # Adding element type (line 225)
    int_217081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 17), list_217079, int_217081)
    
    # Processing the call keyword arguments (line 225)
    kwargs_217082 = {}
    # Getting the type of 'np' (line 225)
    np_217077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 225)
    array_217078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), np_217077, 'array')
    # Calling array(args, kwargs) (line 225)
    array_call_result_217083 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), array_217078, *[list_217079], **kwargs_217082)
    
    # Assigning a type to the variable 'y' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'y', array_call_result_217083)
    
    # Assigning a Call to a Tuple (line 226):
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_217084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to reflective_transformation(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'y' (line 226)
    y_217086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 37), 'y', False)
    # Getting the type of 'lb' (line 226)
    lb_217087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'lb', False)
    # Getting the type of 'ub' (line 226)
    ub_217088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'ub', False)
    # Processing the call keyword arguments (line 226)
    kwargs_217089 = {}
    # Getting the type of 'reflective_transformation' (line 226)
    reflective_transformation_217085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 226)
    reflective_transformation_call_result_217090 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), reflective_transformation_217085, *[y_217086, lb_217087, ub_217088], **kwargs_217089)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___217091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), reflective_transformation_call_result_217090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_217092 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___217091, int_217084)
    
    # Assigning a type to the variable 'tuple_var_assignment_215659' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_215659', subscript_call_result_217092)
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_217093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to reflective_transformation(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'y' (line 226)
    y_217095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 37), 'y', False)
    # Getting the type of 'lb' (line 226)
    lb_217096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'lb', False)
    # Getting the type of 'ub' (line 226)
    ub_217097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'ub', False)
    # Processing the call keyword arguments (line 226)
    kwargs_217098 = {}
    # Getting the type of 'reflective_transformation' (line 226)
    reflective_transformation_217094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 226)
    reflective_transformation_call_result_217099 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), reflective_transformation_217094, *[y_217095, lb_217096, ub_217097], **kwargs_217098)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___217100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), reflective_transformation_call_result_217099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_217101 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___217100, int_217093)
    
    # Assigning a type to the variable 'tuple_var_assignment_215660' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_215660', subscript_call_result_217101)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_215659' (line 226)
    tuple_var_assignment_215659_217102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_215659')
    # Assigning a type to the variable 'x' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'x', tuple_var_assignment_215659_217102)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_215660' (line 226)
    tuple_var_assignment_215660_217103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_215660')
    # Assigning a type to the variable 'g' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'g', tuple_var_assignment_215660_217103)
    
    # Call to assert_equal(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'x' (line 227)
    x_217105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 17), 'x', False)
    # Getting the type of 'y' (line 227)
    y_217106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'y', False)
    # Processing the call keyword arguments (line 227)
    kwargs_217107 = {}
    # Getting the type of 'assert_equal' (line 227)
    assert_equal_217104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 227)
    assert_equal_call_result_217108 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), assert_equal_217104, *[x_217105, y_217106], **kwargs_217107)
    
    
    # Call to assert_equal(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'g' (line 228)
    g_217110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'g', False)
    
    # Call to ones(...): (line 228)
    # Processing the call arguments (line 228)
    int_217113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 28), 'int')
    # Processing the call keyword arguments (line 228)
    kwargs_217114 = {}
    # Getting the type of 'np' (line 228)
    np_217111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'np', False)
    # Obtaining the member 'ones' of a type (line 228)
    ones_217112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 20), np_217111, 'ones')
    # Calling ones(args, kwargs) (line 228)
    ones_call_result_217115 = invoke(stypy.reporting.localization.Localization(__file__, 228, 20), ones_217112, *[int_217113], **kwargs_217114)
    
    # Processing the call keyword arguments (line 228)
    kwargs_217116 = {}
    # Getting the type of 'assert_equal' (line 228)
    assert_equal_217109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 228)
    assert_equal_call_result_217117 = invoke(stypy.reporting.localization.Localization(__file__, 228, 4), assert_equal_217109, *[g_217110, ones_call_result_217115], **kwargs_217116)
    
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to array(...): (line 230)
    # Processing the call arguments (line 230)
    
    # Obtaining an instance of the builtin type 'list' (line 230)
    list_217120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 230)
    # Adding element type (line 230)
    int_217121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), list_217120, int_217121)
    # Adding element type (line 230)
    int_217122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), list_217120, int_217122)
    
    # Processing the call keyword arguments (line 230)
    # Getting the type of 'float' (line 230)
    float_217123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 32), 'float', False)
    keyword_217124 = float_217123
    kwargs_217125 = {'dtype': keyword_217124}
    # Getting the type of 'np' (line 230)
    np_217118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 230)
    array_217119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), np_217118, 'array')
    # Calling array(args, kwargs) (line 230)
    array_call_result_217126 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), array_217119, *[list_217120], **kwargs_217125)
    
    # Assigning a type to the variable 'y' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'y', array_call_result_217126)
    
    # Assigning a Call to a Tuple (line 232):
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_217127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 4), 'int')
    
    # Call to reflective_transformation(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'y' (line 232)
    y_217129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 37), 'y', False)
    # Getting the type of 'lb' (line 232)
    lb_217130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 40), 'lb', False)
    
    # Call to array(...): (line 232)
    # Processing the call arguments (line 232)
    
    # Obtaining an instance of the builtin type 'list' (line 232)
    list_217133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 232)
    # Adding element type (line 232)
    # Getting the type of 'np' (line 232)
    np_217134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 54), 'np', False)
    # Obtaining the member 'inf' of a type (line 232)
    inf_217135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 54), np_217134, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 53), list_217133, inf_217135)
    # Adding element type (line 232)
    # Getting the type of 'np' (line 232)
    np_217136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 62), 'np', False)
    # Obtaining the member 'inf' of a type (line 232)
    inf_217137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 62), np_217136, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 53), list_217133, inf_217137)
    
    # Processing the call keyword arguments (line 232)
    kwargs_217138 = {}
    # Getting the type of 'np' (line 232)
    np_217131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 44), 'np', False)
    # Obtaining the member 'array' of a type (line 232)
    array_217132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 44), np_217131, 'array')
    # Calling array(args, kwargs) (line 232)
    array_call_result_217139 = invoke(stypy.reporting.localization.Localization(__file__, 232, 44), array_217132, *[list_217133], **kwargs_217138)
    
    # Processing the call keyword arguments (line 232)
    kwargs_217140 = {}
    # Getting the type of 'reflective_transformation' (line 232)
    reflective_transformation_217128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 232)
    reflective_transformation_call_result_217141 = invoke(stypy.reporting.localization.Localization(__file__, 232, 11), reflective_transformation_217128, *[y_217129, lb_217130, array_call_result_217139], **kwargs_217140)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___217142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 4), reflective_transformation_call_result_217141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_217143 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), getitem___217142, int_217127)
    
    # Assigning a type to the variable 'tuple_var_assignment_215661' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_215661', subscript_call_result_217143)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_217144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 4), 'int')
    
    # Call to reflective_transformation(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'y' (line 232)
    y_217146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 37), 'y', False)
    # Getting the type of 'lb' (line 232)
    lb_217147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 40), 'lb', False)
    
    # Call to array(...): (line 232)
    # Processing the call arguments (line 232)
    
    # Obtaining an instance of the builtin type 'list' (line 232)
    list_217150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 232)
    # Adding element type (line 232)
    # Getting the type of 'np' (line 232)
    np_217151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 54), 'np', False)
    # Obtaining the member 'inf' of a type (line 232)
    inf_217152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 54), np_217151, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 53), list_217150, inf_217152)
    # Adding element type (line 232)
    # Getting the type of 'np' (line 232)
    np_217153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 62), 'np', False)
    # Obtaining the member 'inf' of a type (line 232)
    inf_217154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 62), np_217153, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 53), list_217150, inf_217154)
    
    # Processing the call keyword arguments (line 232)
    kwargs_217155 = {}
    # Getting the type of 'np' (line 232)
    np_217148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 44), 'np', False)
    # Obtaining the member 'array' of a type (line 232)
    array_217149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 44), np_217148, 'array')
    # Calling array(args, kwargs) (line 232)
    array_call_result_217156 = invoke(stypy.reporting.localization.Localization(__file__, 232, 44), array_217149, *[list_217150], **kwargs_217155)
    
    # Processing the call keyword arguments (line 232)
    kwargs_217157 = {}
    # Getting the type of 'reflective_transformation' (line 232)
    reflective_transformation_217145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 232)
    reflective_transformation_call_result_217158 = invoke(stypy.reporting.localization.Localization(__file__, 232, 11), reflective_transformation_217145, *[y_217146, lb_217147, array_call_result_217156], **kwargs_217157)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___217159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 4), reflective_transformation_call_result_217158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_217160 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), getitem___217159, int_217144)
    
    # Assigning a type to the variable 'tuple_var_assignment_215662' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_215662', subscript_call_result_217160)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_215661' (line 232)
    tuple_var_assignment_215661_217161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_215661')
    # Assigning a type to the variable 'x' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'x', tuple_var_assignment_215661_217161)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_215662' (line 232)
    tuple_var_assignment_215662_217162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_215662')
    # Assigning a type to the variable 'g' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 7), 'g', tuple_var_assignment_215662_217162)
    
    # Call to assert_equal(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'x' (line 233)
    x_217164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 233)
    list_217165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 233)
    # Adding element type (line 233)
    int_217166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 20), list_217165, int_217166)
    # Adding element type (line 233)
    int_217167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 20), list_217165, int_217167)
    
    # Processing the call keyword arguments (line 233)
    kwargs_217168 = {}
    # Getting the type of 'assert_equal' (line 233)
    assert_equal_217163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 233)
    assert_equal_call_result_217169 = invoke(stypy.reporting.localization.Localization(__file__, 233, 4), assert_equal_217163, *[x_217164, list_217165], **kwargs_217168)
    
    
    # Call to assert_equal(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'g' (line 234)
    g_217171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'g', False)
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_217172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    int_217173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 20), list_217172, int_217173)
    # Adding element type (line 234)
    int_217174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 20), list_217172, int_217174)
    
    # Processing the call keyword arguments (line 234)
    kwargs_217175 = {}
    # Getting the type of 'assert_equal' (line 234)
    assert_equal_217170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 234)
    assert_equal_call_result_217176 = invoke(stypy.reporting.localization.Localization(__file__, 234, 4), assert_equal_217170, *[g_217171, list_217172], **kwargs_217175)
    
    
    # Assigning a Call to a Tuple (line 236):
    
    # Assigning a Subscript to a Name (line 236):
    
    # Obtaining the type of the subscript
    int_217177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 4), 'int')
    
    # Call to reflective_transformation(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'y' (line 236)
    y_217179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 37), 'y', False)
    
    # Call to array(...): (line 236)
    # Processing the call arguments (line 236)
    
    # Obtaining an instance of the builtin type 'list' (line 236)
    list_217182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 236)
    # Adding element type (line 236)
    
    # Getting the type of 'np' (line 236)
    np_217183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 51), 'np', False)
    # Obtaining the member 'inf' of a type (line 236)
    inf_217184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 51), np_217183, 'inf')
    # Applying the 'usub' unary operator (line 236)
    result___neg___217185 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 50), 'usub', inf_217184)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), list_217182, result___neg___217185)
    # Adding element type (line 236)
    
    # Getting the type of 'np' (line 236)
    np_217186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 60), 'np', False)
    # Obtaining the member 'inf' of a type (line 236)
    inf_217187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 60), np_217186, 'inf')
    # Applying the 'usub' unary operator (line 236)
    result___neg___217188 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 59), 'usub', inf_217187)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), list_217182, result___neg___217188)
    
    # Processing the call keyword arguments (line 236)
    kwargs_217189 = {}
    # Getting the type of 'np' (line 236)
    np_217180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 40), 'np', False)
    # Obtaining the member 'array' of a type (line 236)
    array_217181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 40), np_217180, 'array')
    # Calling array(args, kwargs) (line 236)
    array_call_result_217190 = invoke(stypy.reporting.localization.Localization(__file__, 236, 40), array_217181, *[list_217182], **kwargs_217189)
    
    # Getting the type of 'ub' (line 236)
    ub_217191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 70), 'ub', False)
    # Processing the call keyword arguments (line 236)
    kwargs_217192 = {}
    # Getting the type of 'reflective_transformation' (line 236)
    reflective_transformation_217178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 236)
    reflective_transformation_call_result_217193 = invoke(stypy.reporting.localization.Localization(__file__, 236, 11), reflective_transformation_217178, *[y_217179, array_call_result_217190, ub_217191], **kwargs_217192)
    
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___217194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 4), reflective_transformation_call_result_217193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_217195 = invoke(stypy.reporting.localization.Localization(__file__, 236, 4), getitem___217194, int_217177)
    
    # Assigning a type to the variable 'tuple_var_assignment_215663' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'tuple_var_assignment_215663', subscript_call_result_217195)
    
    # Assigning a Subscript to a Name (line 236):
    
    # Obtaining the type of the subscript
    int_217196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 4), 'int')
    
    # Call to reflective_transformation(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'y' (line 236)
    y_217198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 37), 'y', False)
    
    # Call to array(...): (line 236)
    # Processing the call arguments (line 236)
    
    # Obtaining an instance of the builtin type 'list' (line 236)
    list_217201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 236)
    # Adding element type (line 236)
    
    # Getting the type of 'np' (line 236)
    np_217202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 51), 'np', False)
    # Obtaining the member 'inf' of a type (line 236)
    inf_217203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 51), np_217202, 'inf')
    # Applying the 'usub' unary operator (line 236)
    result___neg___217204 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 50), 'usub', inf_217203)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), list_217201, result___neg___217204)
    # Adding element type (line 236)
    
    # Getting the type of 'np' (line 236)
    np_217205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 60), 'np', False)
    # Obtaining the member 'inf' of a type (line 236)
    inf_217206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 60), np_217205, 'inf')
    # Applying the 'usub' unary operator (line 236)
    result___neg___217207 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 59), 'usub', inf_217206)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), list_217201, result___neg___217207)
    
    # Processing the call keyword arguments (line 236)
    kwargs_217208 = {}
    # Getting the type of 'np' (line 236)
    np_217199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 40), 'np', False)
    # Obtaining the member 'array' of a type (line 236)
    array_217200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 40), np_217199, 'array')
    # Calling array(args, kwargs) (line 236)
    array_call_result_217209 = invoke(stypy.reporting.localization.Localization(__file__, 236, 40), array_217200, *[list_217201], **kwargs_217208)
    
    # Getting the type of 'ub' (line 236)
    ub_217210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 70), 'ub', False)
    # Processing the call keyword arguments (line 236)
    kwargs_217211 = {}
    # Getting the type of 'reflective_transformation' (line 236)
    reflective_transformation_217197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 236)
    reflective_transformation_call_result_217212 = invoke(stypy.reporting.localization.Localization(__file__, 236, 11), reflective_transformation_217197, *[y_217198, array_call_result_217209, ub_217210], **kwargs_217211)
    
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___217213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 4), reflective_transformation_call_result_217212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_217214 = invoke(stypy.reporting.localization.Localization(__file__, 236, 4), getitem___217213, int_217196)
    
    # Assigning a type to the variable 'tuple_var_assignment_215664' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'tuple_var_assignment_215664', subscript_call_result_217214)
    
    # Assigning a Name to a Name (line 236):
    # Getting the type of 'tuple_var_assignment_215663' (line 236)
    tuple_var_assignment_215663_217215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'tuple_var_assignment_215663')
    # Assigning a type to the variable 'x' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'x', tuple_var_assignment_215663_217215)
    
    # Assigning a Name to a Name (line 236):
    # Getting the type of 'tuple_var_assignment_215664' (line 236)
    tuple_var_assignment_215664_217216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'tuple_var_assignment_215664')
    # Assigning a type to the variable 'g' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 7), 'g', tuple_var_assignment_215664_217216)
    
    # Call to assert_equal(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'x' (line 237)
    x_217218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 237)
    list_217219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 237)
    # Adding element type (line 237)
    int_217220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 20), list_217219, int_217220)
    # Adding element type (line 237)
    int_217221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 20), list_217219, int_217221)
    
    # Processing the call keyword arguments (line 237)
    kwargs_217222 = {}
    # Getting the type of 'assert_equal' (line 237)
    assert_equal_217217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 237)
    assert_equal_call_result_217223 = invoke(stypy.reporting.localization.Localization(__file__, 237, 4), assert_equal_217217, *[x_217218, list_217219], **kwargs_217222)
    
    
    # Call to assert_equal(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'g' (line 238)
    g_217225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 17), 'g', False)
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_217226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    int_217227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 20), list_217226, int_217227)
    # Adding element type (line 238)
    int_217228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 20), list_217226, int_217228)
    
    # Processing the call keyword arguments (line 238)
    kwargs_217229 = {}
    # Getting the type of 'assert_equal' (line 238)
    assert_equal_217224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 238)
    assert_equal_call_result_217230 = invoke(stypy.reporting.localization.Localization(__file__, 238, 4), assert_equal_217224, *[g_217225, list_217226], **kwargs_217229)
    
    
    # Assigning a Call to a Tuple (line 240):
    
    # Assigning a Subscript to a Name (line 240):
    
    # Obtaining the type of the subscript
    int_217231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'int')
    
    # Call to reflective_transformation(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'y' (line 240)
    y_217233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 37), 'y', False)
    # Getting the type of 'lb' (line 240)
    lb_217234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 40), 'lb', False)
    # Getting the type of 'ub' (line 240)
    ub_217235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'ub', False)
    # Processing the call keyword arguments (line 240)
    kwargs_217236 = {}
    # Getting the type of 'reflective_transformation' (line 240)
    reflective_transformation_217232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 240)
    reflective_transformation_call_result_217237 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), reflective_transformation_217232, *[y_217233, lb_217234, ub_217235], **kwargs_217236)
    
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___217238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), reflective_transformation_call_result_217237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_217239 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), getitem___217238, int_217231)
    
    # Assigning a type to the variable 'tuple_var_assignment_215665' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_215665', subscript_call_result_217239)
    
    # Assigning a Subscript to a Name (line 240):
    
    # Obtaining the type of the subscript
    int_217240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'int')
    
    # Call to reflective_transformation(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'y' (line 240)
    y_217242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 37), 'y', False)
    # Getting the type of 'lb' (line 240)
    lb_217243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 40), 'lb', False)
    # Getting the type of 'ub' (line 240)
    ub_217244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'ub', False)
    # Processing the call keyword arguments (line 240)
    kwargs_217245 = {}
    # Getting the type of 'reflective_transformation' (line 240)
    reflective_transformation_217241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 240)
    reflective_transformation_call_result_217246 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), reflective_transformation_217241, *[y_217242, lb_217243, ub_217244], **kwargs_217245)
    
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___217247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), reflective_transformation_call_result_217246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_217248 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), getitem___217247, int_217240)
    
    # Assigning a type to the variable 'tuple_var_assignment_215666' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_215666', subscript_call_result_217248)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_var_assignment_215665' (line 240)
    tuple_var_assignment_215665_217249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_215665')
    # Assigning a type to the variable 'x' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'x', tuple_var_assignment_215665_217249)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_var_assignment_215666' (line 240)
    tuple_var_assignment_215666_217250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_215666')
    # Assigning a type to the variable 'g' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 'g', tuple_var_assignment_215666_217250)
    
    # Call to assert_equal(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'x' (line 241)
    x_217252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 241)
    list_217253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 241)
    # Adding element type (line 241)
    int_217254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 20), list_217253, int_217254)
    # Adding element type (line 241)
    int_217255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 20), list_217253, int_217255)
    
    # Processing the call keyword arguments (line 241)
    kwargs_217256 = {}
    # Getting the type of 'assert_equal' (line 241)
    assert_equal_217251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 241)
    assert_equal_call_result_217257 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), assert_equal_217251, *[x_217252, list_217253], **kwargs_217256)
    
    
    # Call to assert_equal(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'g' (line 242)
    g_217259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 17), 'g', False)
    
    # Obtaining an instance of the builtin type 'list' (line 242)
    list_217260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 242)
    # Adding element type (line 242)
    int_217261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 20), list_217260, int_217261)
    # Adding element type (line 242)
    int_217262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 20), list_217260, int_217262)
    
    # Processing the call keyword arguments (line 242)
    kwargs_217263 = {}
    # Getting the type of 'assert_equal' (line 242)
    assert_equal_217258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 242)
    assert_equal_call_result_217264 = invoke(stypy.reporting.localization.Localization(__file__, 242, 4), assert_equal_217258, *[g_217259, list_217260], **kwargs_217263)
    
    
    # Assigning a Call to a Name (line 244):
    
    # Assigning a Call to a Name (line 244):
    
    # Call to array(...): (line 244)
    # Processing the call arguments (line 244)
    
    # Obtaining an instance of the builtin type 'list' (line 244)
    list_217267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 244)
    # Adding element type (line 244)
    
    # Getting the type of 'np' (line 244)
    np_217268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'np', False)
    # Obtaining the member 'inf' of a type (line 244)
    inf_217269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 20), np_217268, 'inf')
    # Applying the 'usub' unary operator (line 244)
    result___neg___217270 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 19), 'usub', inf_217269)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 18), list_217267, result___neg___217270)
    # Adding element type (line 244)
    int_217271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 18), list_217267, int_217271)
    
    # Processing the call keyword arguments (line 244)
    kwargs_217272 = {}
    # Getting the type of 'np' (line 244)
    np_217265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 244)
    array_217266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 9), np_217265, 'array')
    # Calling array(args, kwargs) (line 244)
    array_call_result_217273 = invoke(stypy.reporting.localization.Localization(__file__, 244, 9), array_217266, *[list_217267], **kwargs_217272)
    
    # Assigning a type to the variable 'lb' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'lb', array_call_result_217273)
    
    # Assigning a Call to a Name (line 245):
    
    # Assigning a Call to a Name (line 245):
    
    # Call to array(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining an instance of the builtin type 'list' (line 245)
    list_217276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 245)
    # Adding element type (line 245)
    int_217277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 18), list_217276, int_217277)
    # Adding element type (line 245)
    # Getting the type of 'np' (line 245)
    np_217278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 22), 'np', False)
    # Obtaining the member 'inf' of a type (line 245)
    inf_217279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 22), np_217278, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 18), list_217276, inf_217279)
    
    # Processing the call keyword arguments (line 245)
    kwargs_217280 = {}
    # Getting the type of 'np' (line 245)
    np_217274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 245)
    array_217275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 9), np_217274, 'array')
    # Calling array(args, kwargs) (line 245)
    array_call_result_217281 = invoke(stypy.reporting.localization.Localization(__file__, 245, 9), array_217275, *[list_217276], **kwargs_217280)
    
    # Assigning a type to the variable 'ub' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'ub', array_call_result_217281)
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to array(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Obtaining an instance of the builtin type 'list' (line 246)
    list_217284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 246)
    # Adding element type (line 246)
    int_217285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 17), list_217284, int_217285)
    # Adding element type (line 246)
    int_217286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 17), list_217284, int_217286)
    
    # Processing the call keyword arguments (line 246)
    # Getting the type of 'float' (line 246)
    float_217287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 33), 'float', False)
    keyword_217288 = float_217287
    kwargs_217289 = {'dtype': keyword_217288}
    # Getting the type of 'np' (line 246)
    np_217282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 246)
    array_217283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), np_217282, 'array')
    # Calling array(args, kwargs) (line 246)
    array_call_result_217290 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), array_217283, *[list_217284], **kwargs_217289)
    
    # Assigning a type to the variable 'y' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'y', array_call_result_217290)
    
    # Assigning a Call to a Tuple (line 247):
    
    # Assigning a Subscript to a Name (line 247):
    
    # Obtaining the type of the subscript
    int_217291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 4), 'int')
    
    # Call to reflective_transformation(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'y' (line 247)
    y_217293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'y', False)
    # Getting the type of 'lb' (line 247)
    lb_217294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 40), 'lb', False)
    # Getting the type of 'ub' (line 247)
    ub_217295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 44), 'ub', False)
    # Processing the call keyword arguments (line 247)
    kwargs_217296 = {}
    # Getting the type of 'reflective_transformation' (line 247)
    reflective_transformation_217292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 247)
    reflective_transformation_call_result_217297 = invoke(stypy.reporting.localization.Localization(__file__, 247, 11), reflective_transformation_217292, *[y_217293, lb_217294, ub_217295], **kwargs_217296)
    
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___217298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 4), reflective_transformation_call_result_217297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_217299 = invoke(stypy.reporting.localization.Localization(__file__, 247, 4), getitem___217298, int_217291)
    
    # Assigning a type to the variable 'tuple_var_assignment_215667' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'tuple_var_assignment_215667', subscript_call_result_217299)
    
    # Assigning a Subscript to a Name (line 247):
    
    # Obtaining the type of the subscript
    int_217300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 4), 'int')
    
    # Call to reflective_transformation(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'y' (line 247)
    y_217302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'y', False)
    # Getting the type of 'lb' (line 247)
    lb_217303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 40), 'lb', False)
    # Getting the type of 'ub' (line 247)
    ub_217304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 44), 'ub', False)
    # Processing the call keyword arguments (line 247)
    kwargs_217305 = {}
    # Getting the type of 'reflective_transformation' (line 247)
    reflective_transformation_217301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 247)
    reflective_transformation_call_result_217306 = invoke(stypy.reporting.localization.Localization(__file__, 247, 11), reflective_transformation_217301, *[y_217302, lb_217303, ub_217304], **kwargs_217305)
    
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___217307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 4), reflective_transformation_call_result_217306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_217308 = invoke(stypy.reporting.localization.Localization(__file__, 247, 4), getitem___217307, int_217300)
    
    # Assigning a type to the variable 'tuple_var_assignment_215668' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'tuple_var_assignment_215668', subscript_call_result_217308)
    
    # Assigning a Name to a Name (line 247):
    # Getting the type of 'tuple_var_assignment_215667' (line 247)
    tuple_var_assignment_215667_217309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'tuple_var_assignment_215667')
    # Assigning a type to the variable 'x' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'x', tuple_var_assignment_215667_217309)
    
    # Assigning a Name to a Name (line 247):
    # Getting the type of 'tuple_var_assignment_215668' (line 247)
    tuple_var_assignment_215668_217310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'tuple_var_assignment_215668')
    # Assigning a type to the variable 'g' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 7), 'g', tuple_var_assignment_215668_217310)
    
    # Call to assert_equal(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'x' (line 248)
    x_217312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 248)
    list_217313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 248)
    # Adding element type (line 248)
    int_217314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 20), list_217313, int_217314)
    # Adding element type (line 248)
    int_217315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 20), list_217313, int_217315)
    
    # Processing the call keyword arguments (line 248)
    kwargs_217316 = {}
    # Getting the type of 'assert_equal' (line 248)
    assert_equal_217311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 248)
    assert_equal_call_result_217317 = invoke(stypy.reporting.localization.Localization(__file__, 248, 4), assert_equal_217311, *[x_217312, list_217313], **kwargs_217316)
    
    
    # Call to assert_equal(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'g' (line 249)
    g_217319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 17), 'g', False)
    
    # Obtaining an instance of the builtin type 'list' (line 249)
    list_217320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 249)
    # Adding element type (line 249)
    int_217321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 20), list_217320, int_217321)
    # Adding element type (line 249)
    int_217322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 20), list_217320, int_217322)
    
    # Processing the call keyword arguments (line 249)
    kwargs_217323 = {}
    # Getting the type of 'assert_equal' (line 249)
    assert_equal_217318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 249)
    assert_equal_call_result_217324 = invoke(stypy.reporting.localization.Localization(__file__, 249, 4), assert_equal_217318, *[g_217319, list_217320], **kwargs_217323)
    
    
    # ################# End of 'test_reflective_transformation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_reflective_transformation' in the type store
    # Getting the type of 'stypy_return_type' (line 221)
    stypy_return_type_217325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_217325)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_reflective_transformation'
    return stypy_return_type_217325

# Assigning a type to the variable 'test_reflective_transformation' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'test_reflective_transformation', test_reflective_transformation)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
