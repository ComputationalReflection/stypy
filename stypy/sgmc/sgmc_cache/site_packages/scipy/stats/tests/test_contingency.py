
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import (assert_equal, assert_array_equal,
5:          assert_array_almost_equal, assert_approx_equal, assert_allclose)
6: from pytest import raises as assert_raises
7: 
8: from scipy.special import xlogy
9: from scipy.stats.contingency import margins, expected_freq, chi2_contingency
10: 
11: 
12: def test_margins():
13:     a = np.array([1])
14:     m = margins(a)
15:     assert_equal(len(m), 1)
16:     m0 = m[0]
17:     assert_array_equal(m0, np.array([1]))
18: 
19:     a = np.array([[1]])
20:     m0, m1 = margins(a)
21:     expected0 = np.array([[1]])
22:     expected1 = np.array([[1]])
23:     assert_array_equal(m0, expected0)
24:     assert_array_equal(m1, expected1)
25: 
26:     a = np.arange(12).reshape(2, 6)
27:     m0, m1 = margins(a)
28:     expected0 = np.array([[15], [51]])
29:     expected1 = np.array([[6, 8, 10, 12, 14, 16]])
30:     assert_array_equal(m0, expected0)
31:     assert_array_equal(m1, expected1)
32: 
33:     a = np.arange(24).reshape(2, 3, 4)
34:     m0, m1, m2 = margins(a)
35:     expected0 = np.array([[[66]], [[210]]])
36:     expected1 = np.array([[[60], [92], [124]]])
37:     expected2 = np.array([[[60, 66, 72, 78]]])
38:     assert_array_equal(m0, expected0)
39:     assert_array_equal(m1, expected1)
40:     assert_array_equal(m2, expected2)
41: 
42: 
43: def test_expected_freq():
44:     assert_array_equal(expected_freq([1]), np.array([1.0]))
45: 
46:     observed = np.array([[[2, 0], [0, 2]], [[0, 2], [2, 0]], [[1, 1], [1, 1]]])
47:     e = expected_freq(observed)
48:     assert_array_equal(e, np.ones_like(observed))
49: 
50:     observed = np.array([[10, 10, 20], [20, 20, 20]])
51:     e = expected_freq(observed)
52:     correct = np.array([[12., 12., 16.], [18., 18., 24.]])
53:     assert_array_almost_equal(e, correct)
54: 
55: 
56: def test_chi2_contingency_trivial():
57:     # Some very simple tests for chi2_contingency.
58: 
59:     # A trivial case
60:     obs = np.array([[1, 2], [1, 2]])
61:     chi2, p, dof, expected = chi2_contingency(obs, correction=False)
62:     assert_equal(chi2, 0.0)
63:     assert_equal(p, 1.0)
64:     assert_equal(dof, 1)
65:     assert_array_equal(obs, expected)
66: 
67:     # A *really* trivial case: 1-D data.
68:     obs = np.array([1, 2, 3])
69:     chi2, p, dof, expected = chi2_contingency(obs, correction=False)
70:     assert_equal(chi2, 0.0)
71:     assert_equal(p, 1.0)
72:     assert_equal(dof, 0)
73:     assert_array_equal(obs, expected)
74: 
75: 
76: def test_chi2_contingency_R():
77:     # Some test cases that were computed independently, using R.
78: 
79:     Rcode = \
80:     '''
81:     # Data vector.
82:     data <- c(
83:       12, 34, 23,     4,  47,  11,
84:       35, 31, 11,    34,  10,  18,
85:       12, 32,  9,    18,  13,  19,
86:       12, 12, 14,     9,  33,  25
87:       )
88: 
89:     # Create factor tags:r=rows, c=columns, t=tiers
90:     r <- factor(gl(4, 2*3, 2*3*4, labels=c("r1", "r2", "r3", "r4")))
91:     c <- factor(gl(3, 1,   2*3*4, labels=c("c1", "c2", "c3")))
92:     t <- factor(gl(2, 3,   2*3*4, labels=c("t1", "t2")))
93: 
94:     # 3-way Chi squared test of independence
95:     s = summary(xtabs(data~r+c+t))
96:     print(s)
97:     '''
98:     Routput = \
99:     '''
100:     Call: xtabs(formula = data ~ r + c + t)
101:     Number of cases in table: 478
102:     Number of factors: 3
103:     Test for independence of all factors:
104:             Chisq = 102.17, df = 17, p-value = 3.514e-14
105:     '''
106:     obs = np.array(
107:         [[[12, 34, 23],
108:           [35, 31, 11],
109:           [12, 32, 9],
110:           [12, 12, 14]],
111:          [[4, 47, 11],
112:           [34, 10, 18],
113:           [18, 13, 19],
114:           [9, 33, 25]]])
115:     chi2, p, dof, expected = chi2_contingency(obs)
116:     assert_approx_equal(chi2, 102.17, significant=5)
117:     assert_approx_equal(p, 3.514e-14, significant=4)
118:     assert_equal(dof, 17)
119: 
120:     Rcode = \
121:     '''
122:     # Data vector.
123:     data <- c(
124:         #
125:         12, 17,
126:         11, 16,
127:         #
128:         11, 12,
129:         15, 16,
130:         #
131:         23, 15,
132:         30, 22,
133:         #
134:         14, 17,
135:         15, 16
136:         )
137: 
138:     # Create factor tags:r=rows, c=columns, d=depths(?), t=tiers
139:     r <- factor(gl(2, 2,  2*2*2*2, labels=c("r1", "r2")))
140:     c <- factor(gl(2, 1,  2*2*2*2, labels=c("c1", "c2")))
141:     d <- factor(gl(2, 4,  2*2*2*2, labels=c("d1", "d2")))
142:     t <- factor(gl(2, 8,  2*2*2*2, labels=c("t1", "t2")))
143: 
144:     # 4-way Chi squared test of independence
145:     s = summary(xtabs(data~r+c+d+t))
146:     print(s)
147:     '''
148:     Routput = \
149:     '''
150:     Call: xtabs(formula = data ~ r + c + d + t)
151:     Number of cases in table: 262
152:     Number of factors: 4
153:     Test for independence of all factors:
154:             Chisq = 8.758, df = 11, p-value = 0.6442
155:     '''
156:     obs = np.array(
157:         [[[[12, 17],
158:            [11, 16]],
159:           [[11, 12],
160:            [15, 16]]],
161:          [[[23, 15],
162:            [30, 22]],
163:           [[14, 17],
164:            [15, 16]]]])
165:     chi2, p, dof, expected = chi2_contingency(obs)
166:     assert_approx_equal(chi2, 8.758, significant=4)
167:     assert_approx_equal(p, 0.6442, significant=4)
168:     assert_equal(dof, 11)
169: 
170: 
171: def test_chi2_contingency_g():
172:     c = np.array([[15, 60], [15, 90]])
173:     g, p, dof, e = chi2_contingency(c, lambda_='log-likelihood', correction=False)
174:     assert_allclose(g, 2*xlogy(c, c/e).sum())
175: 
176:     g, p, dof, e = chi2_contingency(c, lambda_='log-likelihood', correction=True)
177:     c_corr = c + np.array([[-0.5, 0.5], [0.5, -0.5]])
178:     assert_allclose(g, 2*xlogy(c_corr, c_corr/e).sum())
179: 
180:     c = np.array([[10, 12, 10], [12, 10, 10]])
181:     g, p, dof, e = chi2_contingency(c, lambda_='log-likelihood')
182:     assert_allclose(g, 2*xlogy(c, c/e).sum())
183: 
184: 
185: def test_chi2_contingency_bad_args():
186:     # Test that "bad" inputs raise a ValueError.
187: 
188:     # Negative value in the array of observed frequencies.
189:     obs = np.array([[-1, 10], [1, 2]])
190:     assert_raises(ValueError, chi2_contingency, obs)
191: 
192:     # The zeros in this will result in zeros in the array
193:     # of expected frequencies.
194:     obs = np.array([[0, 1], [0, 1]])
195:     assert_raises(ValueError, chi2_contingency, obs)
196: 
197:     # A degenerate case: `observed` has size 0.
198:     obs = np.empty((0, 8))
199:     assert_raises(ValueError, chi2_contingency, obs)
200: 
201: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_632174 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_632174) is not StypyTypeError):

    if (import_632174 != 'pyd_module'):
        __import__(import_632174)
        sys_modules_632175 = sys.modules[import_632174]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_632175.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_632174)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal, assert_approx_equal, assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_632176 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_632176) is not StypyTypeError):

    if (import_632176 != 'pyd_module'):
        __import__(import_632176)
        sys_modules_632177 = sys.modules[import_632176]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_632177.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_array_almost_equal', 'assert_approx_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_632177, sys_modules_632177.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal, assert_approx_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_array_almost_equal', 'assert_approx_equal', 'assert_allclose'], [assert_equal, assert_array_equal, assert_array_almost_equal, assert_approx_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_632176)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from pytest import assert_raises' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_632178 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_632178) is not StypyTypeError):

    if (import_632178 != 'pyd_module'):
        __import__(import_632178)
        sys_modules_632179 = sys.modules[import_632178]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_632179.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_632179, sys_modules_632179.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_632178)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.special import xlogy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_632180 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special')

if (type(import_632180) is not StypyTypeError):

    if (import_632180 != 'pyd_module'):
        __import__(import_632180)
        sys_modules_632181 = sys.modules[import_632180]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', sys_modules_632181.module_type_store, module_type_store, ['xlogy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_632181, sys_modules_632181.module_type_store, module_type_store)
    else:
        from scipy.special import xlogy

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', None, module_type_store, ['xlogy'], [xlogy])

else:
    # Assigning a type to the variable 'scipy.special' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', import_632180)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.stats.contingency import margins, expected_freq, chi2_contingency' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_632182 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.contingency')

if (type(import_632182) is not StypyTypeError):

    if (import_632182 != 'pyd_module'):
        __import__(import_632182)
        sys_modules_632183 = sys.modules[import_632182]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.contingency', sys_modules_632183.module_type_store, module_type_store, ['margins', 'expected_freq', 'chi2_contingency'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_632183, sys_modules_632183.module_type_store, module_type_store)
    else:
        from scipy.stats.contingency import margins, expected_freq, chi2_contingency

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.contingency', None, module_type_store, ['margins', 'expected_freq', 'chi2_contingency'], [margins, expected_freq, chi2_contingency])

else:
    # Assigning a type to the variable 'scipy.stats.contingency' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.contingency', import_632182)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')


@norecursion
def test_margins(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_margins'
    module_type_store = module_type_store.open_function_context('test_margins', 12, 0, False)
    
    # Passed parameters checking function
    test_margins.stypy_localization = localization
    test_margins.stypy_type_of_self = None
    test_margins.stypy_type_store = module_type_store
    test_margins.stypy_function_name = 'test_margins'
    test_margins.stypy_param_names_list = []
    test_margins.stypy_varargs_param_name = None
    test_margins.stypy_kwargs_param_name = None
    test_margins.stypy_call_defaults = defaults
    test_margins.stypy_call_varargs = varargs
    test_margins.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_margins', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_margins', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_margins(...)' code ##################

    
    # Assigning a Call to a Name (line 13):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to array(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_632186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    int_632187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 17), list_632186, int_632187)
    
    # Processing the call keyword arguments (line 13)
    kwargs_632188 = {}
    # Getting the type of 'np' (line 13)
    np_632184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 13)
    array_632185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), np_632184, 'array')
    # Calling array(args, kwargs) (line 13)
    array_call_result_632189 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), array_632185, *[list_632186], **kwargs_632188)
    
    # Assigning a type to the variable 'a' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'a', array_call_result_632189)
    
    # Assigning a Call to a Name (line 14):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to margins(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'a' (line 14)
    a_632191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'a', False)
    # Processing the call keyword arguments (line 14)
    kwargs_632192 = {}
    # Getting the type of 'margins' (line 14)
    margins_632190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'margins', False)
    # Calling margins(args, kwargs) (line 14)
    margins_call_result_632193 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), margins_632190, *[a_632191], **kwargs_632192)
    
    # Assigning a type to the variable 'm' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'm', margins_call_result_632193)
    
    # Call to assert_equal(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to len(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'm' (line 15)
    m_632196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'm', False)
    # Processing the call keyword arguments (line 15)
    kwargs_632197 = {}
    # Getting the type of 'len' (line 15)
    len_632195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'len', False)
    # Calling len(args, kwargs) (line 15)
    len_call_result_632198 = invoke(stypy.reporting.localization.Localization(__file__, 15, 17), len_632195, *[m_632196], **kwargs_632197)
    
    int_632199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'int')
    # Processing the call keyword arguments (line 15)
    kwargs_632200 = {}
    # Getting the type of 'assert_equal' (line 15)
    assert_equal_632194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 15)
    assert_equal_call_result_632201 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), assert_equal_632194, *[len_call_result_632198, int_632199], **kwargs_632200)
    
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_632202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'int')
    # Getting the type of 'm' (line 16)
    m_632203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'm')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___632204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), m_632203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_632205 = invoke(stypy.reporting.localization.Localization(__file__, 16, 9), getitem___632204, int_632202)
    
    # Assigning a type to the variable 'm0' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'm0', subscript_call_result_632205)
    
    # Call to assert_array_equal(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'm0' (line 17)
    m0_632207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'm0', False)
    
    # Call to array(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_632210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_632211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 36), list_632210, int_632211)
    
    # Processing the call keyword arguments (line 17)
    kwargs_632212 = {}
    # Getting the type of 'np' (line 17)
    np_632208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 27), 'np', False)
    # Obtaining the member 'array' of a type (line 17)
    array_632209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 27), np_632208, 'array')
    # Calling array(args, kwargs) (line 17)
    array_call_result_632213 = invoke(stypy.reporting.localization.Localization(__file__, 17, 27), array_632209, *[list_632210], **kwargs_632212)
    
    # Processing the call keyword arguments (line 17)
    kwargs_632214 = {}
    # Getting the type of 'assert_array_equal' (line 17)
    assert_array_equal_632206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 17)
    assert_array_equal_call_result_632215 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), assert_array_equal_632206, *[m0_632207, array_call_result_632213], **kwargs_632214)
    
    
    # Assigning a Call to a Name (line 19):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to array(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_632218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_632219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    int_632220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 18), list_632219, int_632220)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 17), list_632218, list_632219)
    
    # Processing the call keyword arguments (line 19)
    kwargs_632221 = {}
    # Getting the type of 'np' (line 19)
    np_632216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 19)
    array_632217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), np_632216, 'array')
    # Calling array(args, kwargs) (line 19)
    array_call_result_632222 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), array_632217, *[list_632218], **kwargs_632221)
    
    # Assigning a type to the variable 'a' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'a', array_call_result_632222)
    
    # Assigning a Call to a Tuple (line 20):
    
    # Assigning a Subscript to a Name (line 20):
    
    # Obtaining the type of the subscript
    int_632223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'int')
    
    # Call to margins(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'a' (line 20)
    a_632225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'a', False)
    # Processing the call keyword arguments (line 20)
    kwargs_632226 = {}
    # Getting the type of 'margins' (line 20)
    margins_632224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 13), 'margins', False)
    # Calling margins(args, kwargs) (line 20)
    margins_call_result_632227 = invoke(stypy.reporting.localization.Localization(__file__, 20, 13), margins_632224, *[a_632225], **kwargs_632226)
    
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___632228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), margins_call_result_632227, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_632229 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), getitem___632228, int_632223)
    
    # Assigning a type to the variable 'tuple_var_assignment_632139' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple_var_assignment_632139', subscript_call_result_632229)
    
    # Assigning a Subscript to a Name (line 20):
    
    # Obtaining the type of the subscript
    int_632230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'int')
    
    # Call to margins(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'a' (line 20)
    a_632232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'a', False)
    # Processing the call keyword arguments (line 20)
    kwargs_632233 = {}
    # Getting the type of 'margins' (line 20)
    margins_632231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 13), 'margins', False)
    # Calling margins(args, kwargs) (line 20)
    margins_call_result_632234 = invoke(stypy.reporting.localization.Localization(__file__, 20, 13), margins_632231, *[a_632232], **kwargs_632233)
    
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___632235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), margins_call_result_632234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_632236 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), getitem___632235, int_632230)
    
    # Assigning a type to the variable 'tuple_var_assignment_632140' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple_var_assignment_632140', subscript_call_result_632236)
    
    # Assigning a Name to a Name (line 20):
    # Getting the type of 'tuple_var_assignment_632139' (line 20)
    tuple_var_assignment_632139_632237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple_var_assignment_632139')
    # Assigning a type to the variable 'm0' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'm0', tuple_var_assignment_632139_632237)
    
    # Assigning a Name to a Name (line 20):
    # Getting the type of 'tuple_var_assignment_632140' (line 20)
    tuple_var_assignment_632140_632238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple_var_assignment_632140')
    # Assigning a type to the variable 'm1' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'm1', tuple_var_assignment_632140_632238)
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to array(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_632241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_632242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_632243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 26), list_632242, int_632243)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_632241, list_632242)
    
    # Processing the call keyword arguments (line 21)
    kwargs_632244 = {}
    # Getting the type of 'np' (line 21)
    np_632239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 21)
    array_632240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), np_632239, 'array')
    # Calling array(args, kwargs) (line 21)
    array_call_result_632245 = invoke(stypy.reporting.localization.Localization(__file__, 21, 16), array_632240, *[list_632241], **kwargs_632244)
    
    # Assigning a type to the variable 'expected0' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'expected0', array_call_result_632245)
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to array(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_632248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_632249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_632250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 26), list_632249, int_632250)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_632248, list_632249)
    
    # Processing the call keyword arguments (line 22)
    kwargs_632251 = {}
    # Getting the type of 'np' (line 22)
    np_632246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 22)
    array_632247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), np_632246, 'array')
    # Calling array(args, kwargs) (line 22)
    array_call_result_632252 = invoke(stypy.reporting.localization.Localization(__file__, 22, 16), array_632247, *[list_632248], **kwargs_632251)
    
    # Assigning a type to the variable 'expected1' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'expected1', array_call_result_632252)
    
    # Call to assert_array_equal(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'm0' (line 23)
    m0_632254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'm0', False)
    # Getting the type of 'expected0' (line 23)
    expected0_632255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'expected0', False)
    # Processing the call keyword arguments (line 23)
    kwargs_632256 = {}
    # Getting the type of 'assert_array_equal' (line 23)
    assert_array_equal_632253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 23)
    assert_array_equal_call_result_632257 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_array_equal_632253, *[m0_632254, expected0_632255], **kwargs_632256)
    
    
    # Call to assert_array_equal(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'm1' (line 24)
    m1_632259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'm1', False)
    # Getting the type of 'expected1' (line 24)
    expected1_632260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'expected1', False)
    # Processing the call keyword arguments (line 24)
    kwargs_632261 = {}
    # Getting the type of 'assert_array_equal' (line 24)
    assert_array_equal_632258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 24)
    assert_array_equal_call_result_632262 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), assert_array_equal_632258, *[m1_632259, expected1_632260], **kwargs_632261)
    
    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to reshape(...): (line 26)
    # Processing the call arguments (line 26)
    int_632269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'int')
    int_632270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_632271 = {}
    
    # Call to arange(...): (line 26)
    # Processing the call arguments (line 26)
    int_632265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_632266 = {}
    # Getting the type of 'np' (line 26)
    np_632263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 26)
    arange_632264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), np_632263, 'arange')
    # Calling arange(args, kwargs) (line 26)
    arange_call_result_632267 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), arange_632264, *[int_632265], **kwargs_632266)
    
    # Obtaining the member 'reshape' of a type (line 26)
    reshape_632268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), arange_call_result_632267, 'reshape')
    # Calling reshape(args, kwargs) (line 26)
    reshape_call_result_632272 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), reshape_632268, *[int_632269, int_632270], **kwargs_632271)
    
    # Assigning a type to the variable 'a' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'a', reshape_call_result_632272)
    
    # Assigning a Call to a Tuple (line 27):
    
    # Assigning a Subscript to a Name (line 27):
    
    # Obtaining the type of the subscript
    int_632273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'int')
    
    # Call to margins(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'a' (line 27)
    a_632275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'a', False)
    # Processing the call keyword arguments (line 27)
    kwargs_632276 = {}
    # Getting the type of 'margins' (line 27)
    margins_632274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'margins', False)
    # Calling margins(args, kwargs) (line 27)
    margins_call_result_632277 = invoke(stypy.reporting.localization.Localization(__file__, 27, 13), margins_632274, *[a_632275], **kwargs_632276)
    
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___632278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), margins_call_result_632277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_632279 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), getitem___632278, int_632273)
    
    # Assigning a type to the variable 'tuple_var_assignment_632141' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'tuple_var_assignment_632141', subscript_call_result_632279)
    
    # Assigning a Subscript to a Name (line 27):
    
    # Obtaining the type of the subscript
    int_632280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'int')
    
    # Call to margins(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'a' (line 27)
    a_632282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'a', False)
    # Processing the call keyword arguments (line 27)
    kwargs_632283 = {}
    # Getting the type of 'margins' (line 27)
    margins_632281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'margins', False)
    # Calling margins(args, kwargs) (line 27)
    margins_call_result_632284 = invoke(stypy.reporting.localization.Localization(__file__, 27, 13), margins_632281, *[a_632282], **kwargs_632283)
    
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___632285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), margins_call_result_632284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_632286 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), getitem___632285, int_632280)
    
    # Assigning a type to the variable 'tuple_var_assignment_632142' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'tuple_var_assignment_632142', subscript_call_result_632286)
    
    # Assigning a Name to a Name (line 27):
    # Getting the type of 'tuple_var_assignment_632141' (line 27)
    tuple_var_assignment_632141_632287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'tuple_var_assignment_632141')
    # Assigning a type to the variable 'm0' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'm0', tuple_var_assignment_632141_632287)
    
    # Assigning a Name to a Name (line 27):
    # Getting the type of 'tuple_var_assignment_632142' (line 27)
    tuple_var_assignment_632142_632288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'tuple_var_assignment_632142')
    # Assigning a type to the variable 'm1' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'm1', tuple_var_assignment_632142_632288)
    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to array(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_632291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_632292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_632293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 26), list_632292, int_632293)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 25), list_632291, list_632292)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_632294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_632295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 32), list_632294, int_632295)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 25), list_632291, list_632294)
    
    # Processing the call keyword arguments (line 28)
    kwargs_632296 = {}
    # Getting the type of 'np' (line 28)
    np_632289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 28)
    array_632290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), np_632289, 'array')
    # Calling array(args, kwargs) (line 28)
    array_call_result_632297 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), array_632290, *[list_632291], **kwargs_632296)
    
    # Assigning a type to the variable 'expected0' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'expected0', array_call_result_632297)
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to array(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_632300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_632301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    int_632302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), list_632301, int_632302)
    # Adding element type (line 29)
    int_632303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), list_632301, int_632303)
    # Adding element type (line 29)
    int_632304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), list_632301, int_632304)
    # Adding element type (line 29)
    int_632305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), list_632301, int_632305)
    # Adding element type (line 29)
    int_632306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), list_632301, int_632306)
    # Adding element type (line 29)
    int_632307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), list_632301, int_632307)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 25), list_632300, list_632301)
    
    # Processing the call keyword arguments (line 29)
    kwargs_632308 = {}
    # Getting the type of 'np' (line 29)
    np_632298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 29)
    array_632299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 16), np_632298, 'array')
    # Calling array(args, kwargs) (line 29)
    array_call_result_632309 = invoke(stypy.reporting.localization.Localization(__file__, 29, 16), array_632299, *[list_632300], **kwargs_632308)
    
    # Assigning a type to the variable 'expected1' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'expected1', array_call_result_632309)
    
    # Call to assert_array_equal(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'm0' (line 30)
    m0_632311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'm0', False)
    # Getting the type of 'expected0' (line 30)
    expected0_632312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'expected0', False)
    # Processing the call keyword arguments (line 30)
    kwargs_632313 = {}
    # Getting the type of 'assert_array_equal' (line 30)
    assert_array_equal_632310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 30)
    assert_array_equal_call_result_632314 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_array_equal_632310, *[m0_632311, expected0_632312], **kwargs_632313)
    
    
    # Call to assert_array_equal(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'm1' (line 31)
    m1_632316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'm1', False)
    # Getting the type of 'expected1' (line 31)
    expected1_632317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'expected1', False)
    # Processing the call keyword arguments (line 31)
    kwargs_632318 = {}
    # Getting the type of 'assert_array_equal' (line 31)
    assert_array_equal_632315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 31)
    assert_array_equal_call_result_632319 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_array_equal_632315, *[m1_632316, expected1_632317], **kwargs_632318)
    
    
    # Assigning a Call to a Name (line 33):
    
    # Assigning a Call to a Name (line 33):
    
    # Call to reshape(...): (line 33)
    # Processing the call arguments (line 33)
    int_632326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'int')
    int_632327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'int')
    int_632328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 36), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_632329 = {}
    
    # Call to arange(...): (line 33)
    # Processing the call arguments (line 33)
    int_632322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_632323 = {}
    # Getting the type of 'np' (line 33)
    np_632320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 33)
    arange_632321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), np_632320, 'arange')
    # Calling arange(args, kwargs) (line 33)
    arange_call_result_632324 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), arange_632321, *[int_632322], **kwargs_632323)
    
    # Obtaining the member 'reshape' of a type (line 33)
    reshape_632325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), arange_call_result_632324, 'reshape')
    # Calling reshape(args, kwargs) (line 33)
    reshape_call_result_632330 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), reshape_632325, *[int_632326, int_632327, int_632328], **kwargs_632329)
    
    # Assigning a type to the variable 'a' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'a', reshape_call_result_632330)
    
    # Assigning a Call to a Tuple (line 34):
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_632331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'int')
    
    # Call to margins(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'a' (line 34)
    a_632333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'a', False)
    # Processing the call keyword arguments (line 34)
    kwargs_632334 = {}
    # Getting the type of 'margins' (line 34)
    margins_632332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'margins', False)
    # Calling margins(args, kwargs) (line 34)
    margins_call_result_632335 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), margins_632332, *[a_632333], **kwargs_632334)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___632336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), margins_call_result_632335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_632337 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), getitem___632336, int_632331)
    
    # Assigning a type to the variable 'tuple_var_assignment_632143' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_632143', subscript_call_result_632337)
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_632338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'int')
    
    # Call to margins(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'a' (line 34)
    a_632340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'a', False)
    # Processing the call keyword arguments (line 34)
    kwargs_632341 = {}
    # Getting the type of 'margins' (line 34)
    margins_632339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'margins', False)
    # Calling margins(args, kwargs) (line 34)
    margins_call_result_632342 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), margins_632339, *[a_632340], **kwargs_632341)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___632343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), margins_call_result_632342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_632344 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), getitem___632343, int_632338)
    
    # Assigning a type to the variable 'tuple_var_assignment_632144' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_632144', subscript_call_result_632344)
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_632345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'int')
    
    # Call to margins(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'a' (line 34)
    a_632347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'a', False)
    # Processing the call keyword arguments (line 34)
    kwargs_632348 = {}
    # Getting the type of 'margins' (line 34)
    margins_632346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'margins', False)
    # Calling margins(args, kwargs) (line 34)
    margins_call_result_632349 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), margins_632346, *[a_632347], **kwargs_632348)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___632350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), margins_call_result_632349, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_632351 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), getitem___632350, int_632345)
    
    # Assigning a type to the variable 'tuple_var_assignment_632145' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_632145', subscript_call_result_632351)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_632143' (line 34)
    tuple_var_assignment_632143_632352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_632143')
    # Assigning a type to the variable 'm0' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'm0', tuple_var_assignment_632143_632352)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_632144' (line 34)
    tuple_var_assignment_632144_632353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_632144')
    # Assigning a type to the variable 'm1' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'm1', tuple_var_assignment_632144_632353)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_632145' (line 34)
    tuple_var_assignment_632145_632354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_632145')
    # Assigning a type to the variable 'm2' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'm2', tuple_var_assignment_632145_632354)
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to array(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_632357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_632358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_632359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    int_632360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 27), list_632359, int_632360)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 26), list_632358, list_632359)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_632357, list_632358)
    # Adding element type (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_632361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_632362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    int_632363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 35), list_632362, int_632363)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 34), list_632361, list_632362)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_632357, list_632361)
    
    # Processing the call keyword arguments (line 35)
    kwargs_632364 = {}
    # Getting the type of 'np' (line 35)
    np_632355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 35)
    array_632356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), np_632355, 'array')
    # Calling array(args, kwargs) (line 35)
    array_call_result_632365 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), array_632356, *[list_632357], **kwargs_632364)
    
    # Assigning a type to the variable 'expected0' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'expected0', array_call_result_632365)
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to array(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_632368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_632369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_632370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    int_632371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 27), list_632370, int_632371)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 26), list_632369, list_632370)
    # Adding element type (line 36)
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_632372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    int_632373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 33), list_632372, int_632373)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 26), list_632369, list_632372)
    # Adding element type (line 36)
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_632374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    int_632375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 39), list_632374, int_632375)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 26), list_632369, list_632374)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_632368, list_632369)
    
    # Processing the call keyword arguments (line 36)
    kwargs_632376 = {}
    # Getting the type of 'np' (line 36)
    np_632366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 36)
    array_632367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), np_632366, 'array')
    # Calling array(args, kwargs) (line 36)
    array_call_result_632377 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), array_632367, *[list_632368], **kwargs_632376)
    
    # Assigning a type to the variable 'expected1' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'expected1', array_call_result_632377)
    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to array(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_632380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_632381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_632382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    int_632383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 27), list_632382, int_632383)
    # Adding element type (line 37)
    int_632384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 27), list_632382, int_632384)
    # Adding element type (line 37)
    int_632385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 27), list_632382, int_632385)
    # Adding element type (line 37)
    int_632386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 27), list_632382, int_632386)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 26), list_632381, list_632382)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 25), list_632380, list_632381)
    
    # Processing the call keyword arguments (line 37)
    kwargs_632387 = {}
    # Getting the type of 'np' (line 37)
    np_632378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 37)
    array_632379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), np_632378, 'array')
    # Calling array(args, kwargs) (line 37)
    array_call_result_632388 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), array_632379, *[list_632380], **kwargs_632387)
    
    # Assigning a type to the variable 'expected2' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'expected2', array_call_result_632388)
    
    # Call to assert_array_equal(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'm0' (line 38)
    m0_632390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'm0', False)
    # Getting the type of 'expected0' (line 38)
    expected0_632391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'expected0', False)
    # Processing the call keyword arguments (line 38)
    kwargs_632392 = {}
    # Getting the type of 'assert_array_equal' (line 38)
    assert_array_equal_632389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 38)
    assert_array_equal_call_result_632393 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), assert_array_equal_632389, *[m0_632390, expected0_632391], **kwargs_632392)
    
    
    # Call to assert_array_equal(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'm1' (line 39)
    m1_632395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'm1', False)
    # Getting the type of 'expected1' (line 39)
    expected1_632396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'expected1', False)
    # Processing the call keyword arguments (line 39)
    kwargs_632397 = {}
    # Getting the type of 'assert_array_equal' (line 39)
    assert_array_equal_632394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 39)
    assert_array_equal_call_result_632398 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), assert_array_equal_632394, *[m1_632395, expected1_632396], **kwargs_632397)
    
    
    # Call to assert_array_equal(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'm2' (line 40)
    m2_632400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'm2', False)
    # Getting the type of 'expected2' (line 40)
    expected2_632401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'expected2', False)
    # Processing the call keyword arguments (line 40)
    kwargs_632402 = {}
    # Getting the type of 'assert_array_equal' (line 40)
    assert_array_equal_632399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 40)
    assert_array_equal_call_result_632403 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), assert_array_equal_632399, *[m2_632400, expected2_632401], **kwargs_632402)
    
    
    # ################# End of 'test_margins(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_margins' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_632404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_632404)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_margins'
    return stypy_return_type_632404

# Assigning a type to the variable 'test_margins' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'test_margins', test_margins)

@norecursion
def test_expected_freq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_expected_freq'
    module_type_store = module_type_store.open_function_context('test_expected_freq', 43, 0, False)
    
    # Passed parameters checking function
    test_expected_freq.stypy_localization = localization
    test_expected_freq.stypy_type_of_self = None
    test_expected_freq.stypy_type_store = module_type_store
    test_expected_freq.stypy_function_name = 'test_expected_freq'
    test_expected_freq.stypy_param_names_list = []
    test_expected_freq.stypy_varargs_param_name = None
    test_expected_freq.stypy_kwargs_param_name = None
    test_expected_freq.stypy_call_defaults = defaults
    test_expected_freq.stypy_call_varargs = varargs
    test_expected_freq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_expected_freq', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_expected_freq', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_expected_freq(...)' code ##################

    
    # Call to assert_array_equal(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to expected_freq(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_632407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    int_632408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 37), list_632407, int_632408)
    
    # Processing the call keyword arguments (line 44)
    kwargs_632409 = {}
    # Getting the type of 'expected_freq' (line 44)
    expected_freq_632406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'expected_freq', False)
    # Calling expected_freq(args, kwargs) (line 44)
    expected_freq_call_result_632410 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), expected_freq_632406, *[list_632407], **kwargs_632409)
    
    
    # Call to array(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_632413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    float_632414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 53), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 52), list_632413, float_632414)
    
    # Processing the call keyword arguments (line 44)
    kwargs_632415 = {}
    # Getting the type of 'np' (line 44)
    np_632411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 43), 'np', False)
    # Obtaining the member 'array' of a type (line 44)
    array_632412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 43), np_632411, 'array')
    # Calling array(args, kwargs) (line 44)
    array_call_result_632416 = invoke(stypy.reporting.localization.Localization(__file__, 44, 43), array_632412, *[list_632413], **kwargs_632415)
    
    # Processing the call keyword arguments (line 44)
    kwargs_632417 = {}
    # Getting the type of 'assert_array_equal' (line 44)
    assert_array_equal_632405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 44)
    assert_array_equal_call_result_632418 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), assert_array_equal_632405, *[expected_freq_call_result_632410, array_call_result_632416], **kwargs_632417)
    
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to array(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    int_632424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 26), list_632423, int_632424)
    # Adding element type (line 46)
    int_632425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 26), list_632423, int_632425)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 25), list_632422, list_632423)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    int_632427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), list_632426, int_632427)
    # Adding element type (line 46)
    int_632428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), list_632426, int_632428)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 25), list_632422, list_632426)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_632421, list_632422)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    int_632431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 44), list_632430, int_632431)
    # Adding element type (line 46)
    int_632432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 44), list_632430, int_632432)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 43), list_632429, list_632430)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    int_632434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 52), list_632433, int_632434)
    # Adding element type (line 46)
    int_632435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 52), list_632433, int_632435)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 43), list_632429, list_632433)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_632421, list_632429)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 62), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    int_632438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 63), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 62), list_632437, int_632438)
    # Adding element type (line 46)
    int_632439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 62), list_632437, int_632439)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 61), list_632436, list_632437)
    # Adding element type (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_632440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 70), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    int_632441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 71), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 70), list_632440, int_632441)
    # Adding element type (line 46)
    int_632442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 74), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 70), list_632440, int_632442)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 61), list_632436, list_632440)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_632421, list_632436)
    
    # Processing the call keyword arguments (line 46)
    kwargs_632443 = {}
    # Getting the type of 'np' (line 46)
    np_632419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 46)
    array_632420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), np_632419, 'array')
    # Calling array(args, kwargs) (line 46)
    array_call_result_632444 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), array_632420, *[list_632421], **kwargs_632443)
    
    # Assigning a type to the variable 'observed' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'observed', array_call_result_632444)
    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to expected_freq(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'observed' (line 47)
    observed_632446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'observed', False)
    # Processing the call keyword arguments (line 47)
    kwargs_632447 = {}
    # Getting the type of 'expected_freq' (line 47)
    expected_freq_632445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'expected_freq', False)
    # Calling expected_freq(args, kwargs) (line 47)
    expected_freq_call_result_632448 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), expected_freq_632445, *[observed_632446], **kwargs_632447)
    
    # Assigning a type to the variable 'e' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'e', expected_freq_call_result_632448)
    
    # Call to assert_array_equal(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'e' (line 48)
    e_632450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'e', False)
    
    # Call to ones_like(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'observed' (line 48)
    observed_632453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 39), 'observed', False)
    # Processing the call keyword arguments (line 48)
    kwargs_632454 = {}
    # Getting the type of 'np' (line 48)
    np_632451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 48)
    ones_like_632452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 26), np_632451, 'ones_like')
    # Calling ones_like(args, kwargs) (line 48)
    ones_like_call_result_632455 = invoke(stypy.reporting.localization.Localization(__file__, 48, 26), ones_like_632452, *[observed_632453], **kwargs_632454)
    
    # Processing the call keyword arguments (line 48)
    kwargs_632456 = {}
    # Getting the type of 'assert_array_equal' (line 48)
    assert_array_equal_632449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 48)
    assert_array_equal_call_result_632457 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), assert_array_equal_632449, *[e_632450, ones_like_call_result_632455], **kwargs_632456)
    
    
    # Assigning a Call to a Name (line 50):
    
    # Assigning a Call to a Name (line 50):
    
    # Call to array(...): (line 50)
    # Processing the call arguments (line 50)
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_632460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_632461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    int_632462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 25), list_632461, int_632462)
    # Adding element type (line 50)
    int_632463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 25), list_632461, int_632463)
    # Adding element type (line 50)
    int_632464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 25), list_632461, int_632464)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 24), list_632460, list_632461)
    # Adding element type (line 50)
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_632465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    int_632466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 39), list_632465, int_632466)
    # Adding element type (line 50)
    int_632467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 39), list_632465, int_632467)
    # Adding element type (line 50)
    int_632468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 39), list_632465, int_632468)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 24), list_632460, list_632465)
    
    # Processing the call keyword arguments (line 50)
    kwargs_632469 = {}
    # Getting the type of 'np' (line 50)
    np_632458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 50)
    array_632459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), np_632458, 'array')
    # Calling array(args, kwargs) (line 50)
    array_call_result_632470 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), array_632459, *[list_632460], **kwargs_632469)
    
    # Assigning a type to the variable 'observed' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'observed', array_call_result_632470)
    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to expected_freq(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'observed' (line 51)
    observed_632472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'observed', False)
    # Processing the call keyword arguments (line 51)
    kwargs_632473 = {}
    # Getting the type of 'expected_freq' (line 51)
    expected_freq_632471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'expected_freq', False)
    # Calling expected_freq(args, kwargs) (line 51)
    expected_freq_call_result_632474 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), expected_freq_632471, *[observed_632472], **kwargs_632473)
    
    # Assigning a type to the variable 'e' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'e', expected_freq_call_result_632474)
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to array(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_632477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_632478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    float_632479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 24), list_632478, float_632479)
    # Adding element type (line 52)
    float_632480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 24), list_632478, float_632480)
    # Adding element type (line 52)
    float_632481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 24), list_632478, float_632481)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 23), list_632477, list_632478)
    # Adding element type (line 52)
    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_632482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    float_632483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 41), list_632482, float_632483)
    # Adding element type (line 52)
    float_632484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 41), list_632482, float_632484)
    # Adding element type (line 52)
    float_632485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 41), list_632482, float_632485)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 23), list_632477, list_632482)
    
    # Processing the call keyword arguments (line 52)
    kwargs_632486 = {}
    # Getting the type of 'np' (line 52)
    np_632475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 52)
    array_632476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 14), np_632475, 'array')
    # Calling array(args, kwargs) (line 52)
    array_call_result_632487 = invoke(stypy.reporting.localization.Localization(__file__, 52, 14), array_632476, *[list_632477], **kwargs_632486)
    
    # Assigning a type to the variable 'correct' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'correct', array_call_result_632487)
    
    # Call to assert_array_almost_equal(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'e' (line 53)
    e_632489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'e', False)
    # Getting the type of 'correct' (line 53)
    correct_632490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'correct', False)
    # Processing the call keyword arguments (line 53)
    kwargs_632491 = {}
    # Getting the type of 'assert_array_almost_equal' (line 53)
    assert_array_almost_equal_632488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 53)
    assert_array_almost_equal_call_result_632492 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), assert_array_almost_equal_632488, *[e_632489, correct_632490], **kwargs_632491)
    
    
    # ################# End of 'test_expected_freq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_expected_freq' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_632493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_632493)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_expected_freq'
    return stypy_return_type_632493

# Assigning a type to the variable 'test_expected_freq' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'test_expected_freq', test_expected_freq)

@norecursion
def test_chi2_contingency_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_chi2_contingency_trivial'
    module_type_store = module_type_store.open_function_context('test_chi2_contingency_trivial', 56, 0, False)
    
    # Passed parameters checking function
    test_chi2_contingency_trivial.stypy_localization = localization
    test_chi2_contingency_trivial.stypy_type_of_self = None
    test_chi2_contingency_trivial.stypy_type_store = module_type_store
    test_chi2_contingency_trivial.stypy_function_name = 'test_chi2_contingency_trivial'
    test_chi2_contingency_trivial.stypy_param_names_list = []
    test_chi2_contingency_trivial.stypy_varargs_param_name = None
    test_chi2_contingency_trivial.stypy_kwargs_param_name = None
    test_chi2_contingency_trivial.stypy_call_defaults = defaults
    test_chi2_contingency_trivial.stypy_call_varargs = varargs
    test_chi2_contingency_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_chi2_contingency_trivial', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_chi2_contingency_trivial', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_chi2_contingency_trivial(...)' code ##################

    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to array(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_632496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_632497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    int_632498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 20), list_632497, int_632498)
    # Adding element type (line 60)
    int_632499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 20), list_632497, int_632499)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_632496, list_632497)
    # Adding element type (line 60)
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_632500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    int_632501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 28), list_632500, int_632501)
    # Adding element type (line 60)
    int_632502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 28), list_632500, int_632502)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_632496, list_632500)
    
    # Processing the call keyword arguments (line 60)
    kwargs_632503 = {}
    # Getting the type of 'np' (line 60)
    np_632494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 60)
    array_632495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 10), np_632494, 'array')
    # Calling array(args, kwargs) (line 60)
    array_call_result_632504 = invoke(stypy.reporting.localization.Localization(__file__, 60, 10), array_632495, *[list_632496], **kwargs_632503)
    
    # Assigning a type to the variable 'obs' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'obs', array_call_result_632504)
    
    # Assigning a Call to a Tuple (line 61):
    
    # Assigning a Subscript to a Name (line 61):
    
    # Obtaining the type of the subscript
    int_632505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'int')
    
    # Call to chi2_contingency(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'obs' (line 61)
    obs_632507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'obs', False)
    # Processing the call keyword arguments (line 61)
    # Getting the type of 'False' (line 61)
    False_632508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 62), 'False', False)
    keyword_632509 = False_632508
    kwargs_632510 = {'correction': keyword_632509}
    # Getting the type of 'chi2_contingency' (line 61)
    chi2_contingency_632506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 61)
    chi2_contingency_call_result_632511 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), chi2_contingency_632506, *[obs_632507], **kwargs_632510)
    
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___632512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), chi2_contingency_call_result_632511, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_632513 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), getitem___632512, int_632505)
    
    # Assigning a type to the variable 'tuple_var_assignment_632146' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_632146', subscript_call_result_632513)
    
    # Assigning a Subscript to a Name (line 61):
    
    # Obtaining the type of the subscript
    int_632514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'int')
    
    # Call to chi2_contingency(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'obs' (line 61)
    obs_632516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'obs', False)
    # Processing the call keyword arguments (line 61)
    # Getting the type of 'False' (line 61)
    False_632517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 62), 'False', False)
    keyword_632518 = False_632517
    kwargs_632519 = {'correction': keyword_632518}
    # Getting the type of 'chi2_contingency' (line 61)
    chi2_contingency_632515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 61)
    chi2_contingency_call_result_632520 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), chi2_contingency_632515, *[obs_632516], **kwargs_632519)
    
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___632521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), chi2_contingency_call_result_632520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_632522 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), getitem___632521, int_632514)
    
    # Assigning a type to the variable 'tuple_var_assignment_632147' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_632147', subscript_call_result_632522)
    
    # Assigning a Subscript to a Name (line 61):
    
    # Obtaining the type of the subscript
    int_632523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'int')
    
    # Call to chi2_contingency(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'obs' (line 61)
    obs_632525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'obs', False)
    # Processing the call keyword arguments (line 61)
    # Getting the type of 'False' (line 61)
    False_632526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 62), 'False', False)
    keyword_632527 = False_632526
    kwargs_632528 = {'correction': keyword_632527}
    # Getting the type of 'chi2_contingency' (line 61)
    chi2_contingency_632524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 61)
    chi2_contingency_call_result_632529 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), chi2_contingency_632524, *[obs_632525], **kwargs_632528)
    
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___632530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), chi2_contingency_call_result_632529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_632531 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), getitem___632530, int_632523)
    
    # Assigning a type to the variable 'tuple_var_assignment_632148' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_632148', subscript_call_result_632531)
    
    # Assigning a Subscript to a Name (line 61):
    
    # Obtaining the type of the subscript
    int_632532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'int')
    
    # Call to chi2_contingency(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'obs' (line 61)
    obs_632534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'obs', False)
    # Processing the call keyword arguments (line 61)
    # Getting the type of 'False' (line 61)
    False_632535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 62), 'False', False)
    keyword_632536 = False_632535
    kwargs_632537 = {'correction': keyword_632536}
    # Getting the type of 'chi2_contingency' (line 61)
    chi2_contingency_632533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 61)
    chi2_contingency_call_result_632538 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), chi2_contingency_632533, *[obs_632534], **kwargs_632537)
    
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___632539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), chi2_contingency_call_result_632538, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_632540 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), getitem___632539, int_632532)
    
    # Assigning a type to the variable 'tuple_var_assignment_632149' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_632149', subscript_call_result_632540)
    
    # Assigning a Name to a Name (line 61):
    # Getting the type of 'tuple_var_assignment_632146' (line 61)
    tuple_var_assignment_632146_632541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_632146')
    # Assigning a type to the variable 'chi2' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'chi2', tuple_var_assignment_632146_632541)
    
    # Assigning a Name to a Name (line 61):
    # Getting the type of 'tuple_var_assignment_632147' (line 61)
    tuple_var_assignment_632147_632542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_632147')
    # Assigning a type to the variable 'p' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 10), 'p', tuple_var_assignment_632147_632542)
    
    # Assigning a Name to a Name (line 61):
    # Getting the type of 'tuple_var_assignment_632148' (line 61)
    tuple_var_assignment_632148_632543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_632148')
    # Assigning a type to the variable 'dof' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'dof', tuple_var_assignment_632148_632543)
    
    # Assigning a Name to a Name (line 61):
    # Getting the type of 'tuple_var_assignment_632149' (line 61)
    tuple_var_assignment_632149_632544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_632149')
    # Assigning a type to the variable 'expected' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'expected', tuple_var_assignment_632149_632544)
    
    # Call to assert_equal(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'chi2' (line 62)
    chi2_632546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'chi2', False)
    float_632547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'float')
    # Processing the call keyword arguments (line 62)
    kwargs_632548 = {}
    # Getting the type of 'assert_equal' (line 62)
    assert_equal_632545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 62)
    assert_equal_call_result_632549 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), assert_equal_632545, *[chi2_632546, float_632547], **kwargs_632548)
    
    
    # Call to assert_equal(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'p' (line 63)
    p_632551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'p', False)
    float_632552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'float')
    # Processing the call keyword arguments (line 63)
    kwargs_632553 = {}
    # Getting the type of 'assert_equal' (line 63)
    assert_equal_632550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 63)
    assert_equal_call_result_632554 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), assert_equal_632550, *[p_632551, float_632552], **kwargs_632553)
    
    
    # Call to assert_equal(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'dof' (line 64)
    dof_632556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'dof', False)
    int_632557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'int')
    # Processing the call keyword arguments (line 64)
    kwargs_632558 = {}
    # Getting the type of 'assert_equal' (line 64)
    assert_equal_632555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 64)
    assert_equal_call_result_632559 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), assert_equal_632555, *[dof_632556, int_632557], **kwargs_632558)
    
    
    # Call to assert_array_equal(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'obs' (line 65)
    obs_632561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'obs', False)
    # Getting the type of 'expected' (line 65)
    expected_632562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'expected', False)
    # Processing the call keyword arguments (line 65)
    kwargs_632563 = {}
    # Getting the type of 'assert_array_equal' (line 65)
    assert_array_equal_632560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 65)
    assert_array_equal_call_result_632564 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), assert_array_equal_632560, *[obs_632561, expected_632562], **kwargs_632563)
    
    
    # Assigning a Call to a Name (line 68):
    
    # Assigning a Call to a Name (line 68):
    
    # Call to array(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_632567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    int_632568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 19), list_632567, int_632568)
    # Adding element type (line 68)
    int_632569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 19), list_632567, int_632569)
    # Adding element type (line 68)
    int_632570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 19), list_632567, int_632570)
    
    # Processing the call keyword arguments (line 68)
    kwargs_632571 = {}
    # Getting the type of 'np' (line 68)
    np_632565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 68)
    array_632566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 10), np_632565, 'array')
    # Calling array(args, kwargs) (line 68)
    array_call_result_632572 = invoke(stypy.reporting.localization.Localization(__file__, 68, 10), array_632566, *[list_632567], **kwargs_632571)
    
    # Assigning a type to the variable 'obs' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'obs', array_call_result_632572)
    
    # Assigning a Call to a Tuple (line 69):
    
    # Assigning a Subscript to a Name (line 69):
    
    # Obtaining the type of the subscript
    int_632573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'int')
    
    # Call to chi2_contingency(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'obs' (line 69)
    obs_632575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 46), 'obs', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'False' (line 69)
    False_632576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 62), 'False', False)
    keyword_632577 = False_632576
    kwargs_632578 = {'correction': keyword_632577}
    # Getting the type of 'chi2_contingency' (line 69)
    chi2_contingency_632574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 69)
    chi2_contingency_call_result_632579 = invoke(stypy.reporting.localization.Localization(__file__, 69, 29), chi2_contingency_632574, *[obs_632575], **kwargs_632578)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___632580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), chi2_contingency_call_result_632579, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_632581 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), getitem___632580, int_632573)
    
    # Assigning a type to the variable 'tuple_var_assignment_632150' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'tuple_var_assignment_632150', subscript_call_result_632581)
    
    # Assigning a Subscript to a Name (line 69):
    
    # Obtaining the type of the subscript
    int_632582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'int')
    
    # Call to chi2_contingency(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'obs' (line 69)
    obs_632584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 46), 'obs', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'False' (line 69)
    False_632585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 62), 'False', False)
    keyword_632586 = False_632585
    kwargs_632587 = {'correction': keyword_632586}
    # Getting the type of 'chi2_contingency' (line 69)
    chi2_contingency_632583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 69)
    chi2_contingency_call_result_632588 = invoke(stypy.reporting.localization.Localization(__file__, 69, 29), chi2_contingency_632583, *[obs_632584], **kwargs_632587)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___632589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), chi2_contingency_call_result_632588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_632590 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), getitem___632589, int_632582)
    
    # Assigning a type to the variable 'tuple_var_assignment_632151' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'tuple_var_assignment_632151', subscript_call_result_632590)
    
    # Assigning a Subscript to a Name (line 69):
    
    # Obtaining the type of the subscript
    int_632591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'int')
    
    # Call to chi2_contingency(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'obs' (line 69)
    obs_632593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 46), 'obs', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'False' (line 69)
    False_632594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 62), 'False', False)
    keyword_632595 = False_632594
    kwargs_632596 = {'correction': keyword_632595}
    # Getting the type of 'chi2_contingency' (line 69)
    chi2_contingency_632592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 69)
    chi2_contingency_call_result_632597 = invoke(stypy.reporting.localization.Localization(__file__, 69, 29), chi2_contingency_632592, *[obs_632593], **kwargs_632596)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___632598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), chi2_contingency_call_result_632597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_632599 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), getitem___632598, int_632591)
    
    # Assigning a type to the variable 'tuple_var_assignment_632152' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'tuple_var_assignment_632152', subscript_call_result_632599)
    
    # Assigning a Subscript to a Name (line 69):
    
    # Obtaining the type of the subscript
    int_632600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'int')
    
    # Call to chi2_contingency(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'obs' (line 69)
    obs_632602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 46), 'obs', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'False' (line 69)
    False_632603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 62), 'False', False)
    keyword_632604 = False_632603
    kwargs_632605 = {'correction': keyword_632604}
    # Getting the type of 'chi2_contingency' (line 69)
    chi2_contingency_632601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 69)
    chi2_contingency_call_result_632606 = invoke(stypy.reporting.localization.Localization(__file__, 69, 29), chi2_contingency_632601, *[obs_632602], **kwargs_632605)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___632607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), chi2_contingency_call_result_632606, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_632608 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), getitem___632607, int_632600)
    
    # Assigning a type to the variable 'tuple_var_assignment_632153' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'tuple_var_assignment_632153', subscript_call_result_632608)
    
    # Assigning a Name to a Name (line 69):
    # Getting the type of 'tuple_var_assignment_632150' (line 69)
    tuple_var_assignment_632150_632609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'tuple_var_assignment_632150')
    # Assigning a type to the variable 'chi2' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'chi2', tuple_var_assignment_632150_632609)
    
    # Assigning a Name to a Name (line 69):
    # Getting the type of 'tuple_var_assignment_632151' (line 69)
    tuple_var_assignment_632151_632610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'tuple_var_assignment_632151')
    # Assigning a type to the variable 'p' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 10), 'p', tuple_var_assignment_632151_632610)
    
    # Assigning a Name to a Name (line 69):
    # Getting the type of 'tuple_var_assignment_632152' (line 69)
    tuple_var_assignment_632152_632611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'tuple_var_assignment_632152')
    # Assigning a type to the variable 'dof' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'dof', tuple_var_assignment_632152_632611)
    
    # Assigning a Name to a Name (line 69):
    # Getting the type of 'tuple_var_assignment_632153' (line 69)
    tuple_var_assignment_632153_632612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'tuple_var_assignment_632153')
    # Assigning a type to the variable 'expected' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'expected', tuple_var_assignment_632153_632612)
    
    # Call to assert_equal(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'chi2' (line 70)
    chi2_632614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'chi2', False)
    float_632615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'float')
    # Processing the call keyword arguments (line 70)
    kwargs_632616 = {}
    # Getting the type of 'assert_equal' (line 70)
    assert_equal_632613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 70)
    assert_equal_call_result_632617 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), assert_equal_632613, *[chi2_632614, float_632615], **kwargs_632616)
    
    
    # Call to assert_equal(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'p' (line 71)
    p_632619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'p', False)
    float_632620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'float')
    # Processing the call keyword arguments (line 71)
    kwargs_632621 = {}
    # Getting the type of 'assert_equal' (line 71)
    assert_equal_632618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 71)
    assert_equal_call_result_632622 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), assert_equal_632618, *[p_632619, float_632620], **kwargs_632621)
    
    
    # Call to assert_equal(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'dof' (line 72)
    dof_632624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'dof', False)
    int_632625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'int')
    # Processing the call keyword arguments (line 72)
    kwargs_632626 = {}
    # Getting the type of 'assert_equal' (line 72)
    assert_equal_632623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 72)
    assert_equal_call_result_632627 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), assert_equal_632623, *[dof_632624, int_632625], **kwargs_632626)
    
    
    # Call to assert_array_equal(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'obs' (line 73)
    obs_632629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'obs', False)
    # Getting the type of 'expected' (line 73)
    expected_632630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'expected', False)
    # Processing the call keyword arguments (line 73)
    kwargs_632631 = {}
    # Getting the type of 'assert_array_equal' (line 73)
    assert_array_equal_632628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 73)
    assert_array_equal_call_result_632632 = invoke(stypy.reporting.localization.Localization(__file__, 73, 4), assert_array_equal_632628, *[obs_632629, expected_632630], **kwargs_632631)
    
    
    # ################# End of 'test_chi2_contingency_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_chi2_contingency_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_632633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_632633)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_chi2_contingency_trivial'
    return stypy_return_type_632633

# Assigning a type to the variable 'test_chi2_contingency_trivial' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'test_chi2_contingency_trivial', test_chi2_contingency_trivial)

@norecursion
def test_chi2_contingency_R(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_chi2_contingency_R'
    module_type_store = module_type_store.open_function_context('test_chi2_contingency_R', 76, 0, False)
    
    # Passed parameters checking function
    test_chi2_contingency_R.stypy_localization = localization
    test_chi2_contingency_R.stypy_type_of_self = None
    test_chi2_contingency_R.stypy_type_store = module_type_store
    test_chi2_contingency_R.stypy_function_name = 'test_chi2_contingency_R'
    test_chi2_contingency_R.stypy_param_names_list = []
    test_chi2_contingency_R.stypy_varargs_param_name = None
    test_chi2_contingency_R.stypy_kwargs_param_name = None
    test_chi2_contingency_R.stypy_call_defaults = defaults
    test_chi2_contingency_R.stypy_call_varargs = varargs
    test_chi2_contingency_R.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_chi2_contingency_R', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_chi2_contingency_R', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_chi2_contingency_R(...)' code ##################

    
    # Assigning a Str to a Name (line 79):
    
    # Assigning a Str to a Name (line 79):
    str_632634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', '\n    # Data vector.\n    data <- c(\n      12, 34, 23,     4,  47,  11,\n      35, 31, 11,    34,  10,  18,\n      12, 32,  9,    18,  13,  19,\n      12, 12, 14,     9,  33,  25\n      )\n\n    # Create factor tags:r=rows, c=columns, t=tiers\n    r <- factor(gl(4, 2*3, 2*3*4, labels=c("r1", "r2", "r3", "r4")))\n    c <- factor(gl(3, 1,   2*3*4, labels=c("c1", "c2", "c3")))\n    t <- factor(gl(2, 3,   2*3*4, labels=c("t1", "t2")))\n\n    # 3-way Chi squared test of independence\n    s = summary(xtabs(data~r+c+t))\n    print(s)\n    ')
    # Assigning a type to the variable 'Rcode' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'Rcode', str_632634)
    
    # Assigning a Str to a Name (line 98):
    
    # Assigning a Str to a Name (line 98):
    str_632635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, (-1)), 'str', '\n    Call: xtabs(formula = data ~ r + c + t)\n    Number of cases in table: 478\n    Number of factors: 3\n    Test for independence of all factors:\n            Chisq = 102.17, df = 17, p-value = 3.514e-14\n    ')
    # Assigning a type to the variable 'Routput' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'Routput', str_632635)
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to array(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_632638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_632639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_632640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    # Adding element type (line 107)
    int_632641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 10), list_632640, int_632641)
    # Adding element type (line 107)
    int_632642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 10), list_632640, int_632642)
    # Adding element type (line 107)
    int_632643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 10), list_632640, int_632643)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 9), list_632639, list_632640)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 108)
    list_632644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 108)
    # Adding element type (line 108)
    int_632645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 10), list_632644, int_632645)
    # Adding element type (line 108)
    int_632646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 10), list_632644, int_632646)
    # Adding element type (line 108)
    int_632647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 10), list_632644, int_632647)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 9), list_632639, list_632644)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_632648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    int_632649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 10), list_632648, int_632649)
    # Adding element type (line 109)
    int_632650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 10), list_632648, int_632650)
    # Adding element type (line 109)
    int_632651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 10), list_632648, int_632651)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 9), list_632639, list_632648)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_632652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    # Adding element type (line 110)
    int_632653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 10), list_632652, int_632653)
    # Adding element type (line 110)
    int_632654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 10), list_632652, int_632654)
    # Adding element type (line 110)
    int_632655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 10), list_632652, int_632655)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 9), list_632639, list_632652)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 8), list_632638, list_632639)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 111)
    list_632656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 111)
    # Adding element type (line 111)
    
    # Obtaining an instance of the builtin type 'list' (line 111)
    list_632657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 111)
    # Adding element type (line 111)
    int_632658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 10), list_632657, int_632658)
    # Adding element type (line 111)
    int_632659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 10), list_632657, int_632659)
    # Adding element type (line 111)
    int_632660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 10), list_632657, int_632660)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), list_632656, list_632657)
    # Adding element type (line 111)
    
    # Obtaining an instance of the builtin type 'list' (line 112)
    list_632661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 112)
    # Adding element type (line 112)
    int_632662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 10), list_632661, int_632662)
    # Adding element type (line 112)
    int_632663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 10), list_632661, int_632663)
    # Adding element type (line 112)
    int_632664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 10), list_632661, int_632664)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), list_632656, list_632661)
    # Adding element type (line 111)
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_632665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    # Adding element type (line 113)
    int_632666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 10), list_632665, int_632666)
    # Adding element type (line 113)
    int_632667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 10), list_632665, int_632667)
    # Adding element type (line 113)
    int_632668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 10), list_632665, int_632668)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), list_632656, list_632665)
    # Adding element type (line 111)
    
    # Obtaining an instance of the builtin type 'list' (line 114)
    list_632669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 114)
    # Adding element type (line 114)
    int_632670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 10), list_632669, int_632670)
    # Adding element type (line 114)
    int_632671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 10), list_632669, int_632671)
    # Adding element type (line 114)
    int_632672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 10), list_632669, int_632672)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), list_632656, list_632669)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 8), list_632638, list_632656)
    
    # Processing the call keyword arguments (line 106)
    kwargs_632673 = {}
    # Getting the type of 'np' (line 106)
    np_632636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 106)
    array_632637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 10), np_632636, 'array')
    # Calling array(args, kwargs) (line 106)
    array_call_result_632674 = invoke(stypy.reporting.localization.Localization(__file__, 106, 10), array_632637, *[list_632638], **kwargs_632673)
    
    # Assigning a type to the variable 'obs' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'obs', array_call_result_632674)
    
    # Assigning a Call to a Tuple (line 115):
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_632675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'int')
    
    # Call to chi2_contingency(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'obs' (line 115)
    obs_632677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 46), 'obs', False)
    # Processing the call keyword arguments (line 115)
    kwargs_632678 = {}
    # Getting the type of 'chi2_contingency' (line 115)
    chi2_contingency_632676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 115)
    chi2_contingency_call_result_632679 = invoke(stypy.reporting.localization.Localization(__file__, 115, 29), chi2_contingency_632676, *[obs_632677], **kwargs_632678)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___632680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), chi2_contingency_call_result_632679, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_632681 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), getitem___632680, int_632675)
    
    # Assigning a type to the variable 'tuple_var_assignment_632154' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_632154', subscript_call_result_632681)
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_632682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'int')
    
    # Call to chi2_contingency(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'obs' (line 115)
    obs_632684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 46), 'obs', False)
    # Processing the call keyword arguments (line 115)
    kwargs_632685 = {}
    # Getting the type of 'chi2_contingency' (line 115)
    chi2_contingency_632683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 115)
    chi2_contingency_call_result_632686 = invoke(stypy.reporting.localization.Localization(__file__, 115, 29), chi2_contingency_632683, *[obs_632684], **kwargs_632685)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___632687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), chi2_contingency_call_result_632686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_632688 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), getitem___632687, int_632682)
    
    # Assigning a type to the variable 'tuple_var_assignment_632155' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_632155', subscript_call_result_632688)
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_632689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'int')
    
    # Call to chi2_contingency(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'obs' (line 115)
    obs_632691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 46), 'obs', False)
    # Processing the call keyword arguments (line 115)
    kwargs_632692 = {}
    # Getting the type of 'chi2_contingency' (line 115)
    chi2_contingency_632690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 115)
    chi2_contingency_call_result_632693 = invoke(stypy.reporting.localization.Localization(__file__, 115, 29), chi2_contingency_632690, *[obs_632691], **kwargs_632692)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___632694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), chi2_contingency_call_result_632693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_632695 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), getitem___632694, int_632689)
    
    # Assigning a type to the variable 'tuple_var_assignment_632156' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_632156', subscript_call_result_632695)
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_632696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'int')
    
    # Call to chi2_contingency(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'obs' (line 115)
    obs_632698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 46), 'obs', False)
    # Processing the call keyword arguments (line 115)
    kwargs_632699 = {}
    # Getting the type of 'chi2_contingency' (line 115)
    chi2_contingency_632697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 115)
    chi2_contingency_call_result_632700 = invoke(stypy.reporting.localization.Localization(__file__, 115, 29), chi2_contingency_632697, *[obs_632698], **kwargs_632699)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___632701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), chi2_contingency_call_result_632700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_632702 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), getitem___632701, int_632696)
    
    # Assigning a type to the variable 'tuple_var_assignment_632157' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_632157', subscript_call_result_632702)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'tuple_var_assignment_632154' (line 115)
    tuple_var_assignment_632154_632703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_632154')
    # Assigning a type to the variable 'chi2' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'chi2', tuple_var_assignment_632154_632703)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'tuple_var_assignment_632155' (line 115)
    tuple_var_assignment_632155_632704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_632155')
    # Assigning a type to the variable 'p' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 10), 'p', tuple_var_assignment_632155_632704)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'tuple_var_assignment_632156' (line 115)
    tuple_var_assignment_632156_632705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_632156')
    # Assigning a type to the variable 'dof' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'dof', tuple_var_assignment_632156_632705)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'tuple_var_assignment_632157' (line 115)
    tuple_var_assignment_632157_632706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_632157')
    # Assigning a type to the variable 'expected' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'expected', tuple_var_assignment_632157_632706)
    
    # Call to assert_approx_equal(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'chi2' (line 116)
    chi2_632708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'chi2', False)
    float_632709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 30), 'float')
    # Processing the call keyword arguments (line 116)
    int_632710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 50), 'int')
    keyword_632711 = int_632710
    kwargs_632712 = {'significant': keyword_632711}
    # Getting the type of 'assert_approx_equal' (line 116)
    assert_approx_equal_632707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'assert_approx_equal', False)
    # Calling assert_approx_equal(args, kwargs) (line 116)
    assert_approx_equal_call_result_632713 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), assert_approx_equal_632707, *[chi2_632708, float_632709], **kwargs_632712)
    
    
    # Call to assert_approx_equal(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'p' (line 117)
    p_632715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'p', False)
    float_632716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'float')
    # Processing the call keyword arguments (line 117)
    int_632717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 50), 'int')
    keyword_632718 = int_632717
    kwargs_632719 = {'significant': keyword_632718}
    # Getting the type of 'assert_approx_equal' (line 117)
    assert_approx_equal_632714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'assert_approx_equal', False)
    # Calling assert_approx_equal(args, kwargs) (line 117)
    assert_approx_equal_call_result_632720 = invoke(stypy.reporting.localization.Localization(__file__, 117, 4), assert_approx_equal_632714, *[p_632715, float_632716], **kwargs_632719)
    
    
    # Call to assert_equal(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'dof' (line 118)
    dof_632722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'dof', False)
    int_632723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 22), 'int')
    # Processing the call keyword arguments (line 118)
    kwargs_632724 = {}
    # Getting the type of 'assert_equal' (line 118)
    assert_equal_632721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 118)
    assert_equal_call_result_632725 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), assert_equal_632721, *[dof_632722, int_632723], **kwargs_632724)
    
    
    # Assigning a Str to a Name (line 120):
    
    # Assigning a Str to a Name (line 120):
    str_632726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'str', '\n    # Data vector.\n    data <- c(\n        #\n        12, 17,\n        11, 16,\n        #\n        11, 12,\n        15, 16,\n        #\n        23, 15,\n        30, 22,\n        #\n        14, 17,\n        15, 16\n        )\n\n    # Create factor tags:r=rows, c=columns, d=depths(?), t=tiers\n    r <- factor(gl(2, 2,  2*2*2*2, labels=c("r1", "r2")))\n    c <- factor(gl(2, 1,  2*2*2*2, labels=c("c1", "c2")))\n    d <- factor(gl(2, 4,  2*2*2*2, labels=c("d1", "d2")))\n    t <- factor(gl(2, 8,  2*2*2*2, labels=c("t1", "t2")))\n\n    # 4-way Chi squared test of independence\n    s = summary(xtabs(data~r+c+d+t))\n    print(s)\n    ')
    # Assigning a type to the variable 'Rcode' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'Rcode', str_632726)
    
    # Assigning a Str to a Name (line 148):
    
    # Assigning a Str to a Name (line 148):
    str_632727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, (-1)), 'str', '\n    Call: xtabs(formula = data ~ r + c + d + t)\n    Number of cases in table: 262\n    Number of factors: 4\n    Test for independence of all factors:\n            Chisq = 8.758, df = 11, p-value = 0.6442\n    ')
    # Assigning a type to the variable 'Routput' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'Routput', str_632727)
    
    # Assigning a Call to a Name (line 156):
    
    # Assigning a Call to a Name (line 156):
    
    # Call to array(...): (line 156)
    # Processing the call arguments (line 156)
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_632730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    # Adding element type (line 157)
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_632731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    # Adding element type (line 157)
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_632732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    # Adding element type (line 157)
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_632733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    # Adding element type (line 157)
    int_632734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 11), list_632733, int_632734)
    # Adding element type (line 157)
    int_632735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 11), list_632733, int_632735)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 10), list_632732, list_632733)
    # Adding element type (line 157)
    
    # Obtaining an instance of the builtin type 'list' (line 158)
    list_632736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 158)
    # Adding element type (line 158)
    int_632737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 11), list_632736, int_632737)
    # Adding element type (line 158)
    int_632738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 11), list_632736, int_632738)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 10), list_632732, list_632736)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 9), list_632731, list_632732)
    # Adding element type (line 157)
    
    # Obtaining an instance of the builtin type 'list' (line 159)
    list_632739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 159)
    # Adding element type (line 159)
    
    # Obtaining an instance of the builtin type 'list' (line 159)
    list_632740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 159)
    # Adding element type (line 159)
    int_632741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 11), list_632740, int_632741)
    # Adding element type (line 159)
    int_632742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 11), list_632740, int_632742)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 10), list_632739, list_632740)
    # Adding element type (line 159)
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_632743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    # Adding element type (line 160)
    int_632744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 11), list_632743, int_632744)
    # Adding element type (line 160)
    int_632745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 11), list_632743, int_632745)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 10), list_632739, list_632743)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 9), list_632731, list_632739)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 8), list_632730, list_632731)
    # Adding element type (line 157)
    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_632746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_632747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_632748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    # Adding element type (line 161)
    int_632749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 11), list_632748, int_632749)
    # Adding element type (line 161)
    int_632750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 11), list_632748, int_632750)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 10), list_632747, list_632748)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'list' (line 162)
    list_632751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 162)
    # Adding element type (line 162)
    int_632752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 11), list_632751, int_632752)
    # Adding element type (line 162)
    int_632753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 11), list_632751, int_632753)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 10), list_632747, list_632751)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 9), list_632746, list_632747)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'list' (line 163)
    list_632754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 163)
    # Adding element type (line 163)
    
    # Obtaining an instance of the builtin type 'list' (line 163)
    list_632755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 163)
    # Adding element type (line 163)
    int_632756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 11), list_632755, int_632756)
    # Adding element type (line 163)
    int_632757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 11), list_632755, int_632757)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 10), list_632754, list_632755)
    # Adding element type (line 163)
    
    # Obtaining an instance of the builtin type 'list' (line 164)
    list_632758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 164)
    # Adding element type (line 164)
    int_632759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 11), list_632758, int_632759)
    # Adding element type (line 164)
    int_632760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 11), list_632758, int_632760)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 10), list_632754, list_632758)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 9), list_632746, list_632754)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 8), list_632730, list_632746)
    
    # Processing the call keyword arguments (line 156)
    kwargs_632761 = {}
    # Getting the type of 'np' (line 156)
    np_632728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 156)
    array_632729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 10), np_632728, 'array')
    # Calling array(args, kwargs) (line 156)
    array_call_result_632762 = invoke(stypy.reporting.localization.Localization(__file__, 156, 10), array_632729, *[list_632730], **kwargs_632761)
    
    # Assigning a type to the variable 'obs' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'obs', array_call_result_632762)
    
    # Assigning a Call to a Tuple (line 165):
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_632763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to chi2_contingency(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'obs' (line 165)
    obs_632765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 46), 'obs', False)
    # Processing the call keyword arguments (line 165)
    kwargs_632766 = {}
    # Getting the type of 'chi2_contingency' (line 165)
    chi2_contingency_632764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 165)
    chi2_contingency_call_result_632767 = invoke(stypy.reporting.localization.Localization(__file__, 165, 29), chi2_contingency_632764, *[obs_632765], **kwargs_632766)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___632768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), chi2_contingency_call_result_632767, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_632769 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___632768, int_632763)
    
    # Assigning a type to the variable 'tuple_var_assignment_632158' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_632158', subscript_call_result_632769)
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_632770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to chi2_contingency(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'obs' (line 165)
    obs_632772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 46), 'obs', False)
    # Processing the call keyword arguments (line 165)
    kwargs_632773 = {}
    # Getting the type of 'chi2_contingency' (line 165)
    chi2_contingency_632771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 165)
    chi2_contingency_call_result_632774 = invoke(stypy.reporting.localization.Localization(__file__, 165, 29), chi2_contingency_632771, *[obs_632772], **kwargs_632773)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___632775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), chi2_contingency_call_result_632774, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_632776 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___632775, int_632770)
    
    # Assigning a type to the variable 'tuple_var_assignment_632159' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_632159', subscript_call_result_632776)
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_632777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to chi2_contingency(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'obs' (line 165)
    obs_632779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 46), 'obs', False)
    # Processing the call keyword arguments (line 165)
    kwargs_632780 = {}
    # Getting the type of 'chi2_contingency' (line 165)
    chi2_contingency_632778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 165)
    chi2_contingency_call_result_632781 = invoke(stypy.reporting.localization.Localization(__file__, 165, 29), chi2_contingency_632778, *[obs_632779], **kwargs_632780)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___632782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), chi2_contingency_call_result_632781, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_632783 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___632782, int_632777)
    
    # Assigning a type to the variable 'tuple_var_assignment_632160' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_632160', subscript_call_result_632783)
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_632784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to chi2_contingency(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'obs' (line 165)
    obs_632786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 46), 'obs', False)
    # Processing the call keyword arguments (line 165)
    kwargs_632787 = {}
    # Getting the type of 'chi2_contingency' (line 165)
    chi2_contingency_632785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 29), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 165)
    chi2_contingency_call_result_632788 = invoke(stypy.reporting.localization.Localization(__file__, 165, 29), chi2_contingency_632785, *[obs_632786], **kwargs_632787)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___632789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), chi2_contingency_call_result_632788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_632790 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___632789, int_632784)
    
    # Assigning a type to the variable 'tuple_var_assignment_632161' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_632161', subscript_call_result_632790)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_632158' (line 165)
    tuple_var_assignment_632158_632791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_632158')
    # Assigning a type to the variable 'chi2' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'chi2', tuple_var_assignment_632158_632791)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_632159' (line 165)
    tuple_var_assignment_632159_632792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_632159')
    # Assigning a type to the variable 'p' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 10), 'p', tuple_var_assignment_632159_632792)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_632160' (line 165)
    tuple_var_assignment_632160_632793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_632160')
    # Assigning a type to the variable 'dof' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'dof', tuple_var_assignment_632160_632793)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_632161' (line 165)
    tuple_var_assignment_632161_632794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_632161')
    # Assigning a type to the variable 'expected' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'expected', tuple_var_assignment_632161_632794)
    
    # Call to assert_approx_equal(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'chi2' (line 166)
    chi2_632796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'chi2', False)
    float_632797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'float')
    # Processing the call keyword arguments (line 166)
    int_632798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 49), 'int')
    keyword_632799 = int_632798
    kwargs_632800 = {'significant': keyword_632799}
    # Getting the type of 'assert_approx_equal' (line 166)
    assert_approx_equal_632795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'assert_approx_equal', False)
    # Calling assert_approx_equal(args, kwargs) (line 166)
    assert_approx_equal_call_result_632801 = invoke(stypy.reporting.localization.Localization(__file__, 166, 4), assert_approx_equal_632795, *[chi2_632796, float_632797], **kwargs_632800)
    
    
    # Call to assert_approx_equal(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'p' (line 167)
    p_632803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'p', False)
    float_632804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'float')
    # Processing the call keyword arguments (line 167)
    int_632805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 47), 'int')
    keyword_632806 = int_632805
    kwargs_632807 = {'significant': keyword_632806}
    # Getting the type of 'assert_approx_equal' (line 167)
    assert_approx_equal_632802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'assert_approx_equal', False)
    # Calling assert_approx_equal(args, kwargs) (line 167)
    assert_approx_equal_call_result_632808 = invoke(stypy.reporting.localization.Localization(__file__, 167, 4), assert_approx_equal_632802, *[p_632803, float_632804], **kwargs_632807)
    
    
    # Call to assert_equal(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'dof' (line 168)
    dof_632810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'dof', False)
    int_632811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 22), 'int')
    # Processing the call keyword arguments (line 168)
    kwargs_632812 = {}
    # Getting the type of 'assert_equal' (line 168)
    assert_equal_632809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 168)
    assert_equal_call_result_632813 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), assert_equal_632809, *[dof_632810, int_632811], **kwargs_632812)
    
    
    # ################# End of 'test_chi2_contingency_R(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_chi2_contingency_R' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_632814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_632814)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_chi2_contingency_R'
    return stypy_return_type_632814

# Assigning a type to the variable 'test_chi2_contingency_R' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'test_chi2_contingency_R', test_chi2_contingency_R)

@norecursion
def test_chi2_contingency_g(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_chi2_contingency_g'
    module_type_store = module_type_store.open_function_context('test_chi2_contingency_g', 171, 0, False)
    
    # Passed parameters checking function
    test_chi2_contingency_g.stypy_localization = localization
    test_chi2_contingency_g.stypy_type_of_self = None
    test_chi2_contingency_g.stypy_type_store = module_type_store
    test_chi2_contingency_g.stypy_function_name = 'test_chi2_contingency_g'
    test_chi2_contingency_g.stypy_param_names_list = []
    test_chi2_contingency_g.stypy_varargs_param_name = None
    test_chi2_contingency_g.stypy_kwargs_param_name = None
    test_chi2_contingency_g.stypy_call_defaults = defaults
    test_chi2_contingency_g.stypy_call_varargs = varargs
    test_chi2_contingency_g.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_chi2_contingency_g', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_chi2_contingency_g', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_chi2_contingency_g(...)' code ##################

    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to array(...): (line 172)
    # Processing the call arguments (line 172)
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_632817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    # Adding element type (line 172)
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_632818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    # Adding element type (line 172)
    int_632819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 18), list_632818, int_632819)
    # Adding element type (line 172)
    int_632820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 18), list_632818, int_632820)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 17), list_632817, list_632818)
    # Adding element type (line 172)
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_632821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    # Adding element type (line 172)
    int_632822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 28), list_632821, int_632822)
    # Adding element type (line 172)
    int_632823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 28), list_632821, int_632823)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 17), list_632817, list_632821)
    
    # Processing the call keyword arguments (line 172)
    kwargs_632824 = {}
    # Getting the type of 'np' (line 172)
    np_632815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 172)
    array_632816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), np_632815, 'array')
    # Calling array(args, kwargs) (line 172)
    array_call_result_632825 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), array_632816, *[list_632817], **kwargs_632824)
    
    # Assigning a type to the variable 'c' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'c', array_call_result_632825)
    
    # Assigning a Call to a Tuple (line 173):
    
    # Assigning a Subscript to a Name (line 173):
    
    # Obtaining the type of the subscript
    int_632826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'int')
    
    # Call to chi2_contingency(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'c' (line 173)
    c_632828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 36), 'c', False)
    # Processing the call keyword arguments (line 173)
    str_632829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 47), 'str', 'log-likelihood')
    keyword_632830 = str_632829
    # Getting the type of 'False' (line 173)
    False_632831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 76), 'False', False)
    keyword_632832 = False_632831
    kwargs_632833 = {'correction': keyword_632832, 'lambda_': keyword_632830}
    # Getting the type of 'chi2_contingency' (line 173)
    chi2_contingency_632827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 173)
    chi2_contingency_call_result_632834 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), chi2_contingency_632827, *[c_632828], **kwargs_632833)
    
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___632835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), chi2_contingency_call_result_632834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_632836 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), getitem___632835, int_632826)
    
    # Assigning a type to the variable 'tuple_var_assignment_632162' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'tuple_var_assignment_632162', subscript_call_result_632836)
    
    # Assigning a Subscript to a Name (line 173):
    
    # Obtaining the type of the subscript
    int_632837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'int')
    
    # Call to chi2_contingency(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'c' (line 173)
    c_632839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 36), 'c', False)
    # Processing the call keyword arguments (line 173)
    str_632840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 47), 'str', 'log-likelihood')
    keyword_632841 = str_632840
    # Getting the type of 'False' (line 173)
    False_632842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 76), 'False', False)
    keyword_632843 = False_632842
    kwargs_632844 = {'correction': keyword_632843, 'lambda_': keyword_632841}
    # Getting the type of 'chi2_contingency' (line 173)
    chi2_contingency_632838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 173)
    chi2_contingency_call_result_632845 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), chi2_contingency_632838, *[c_632839], **kwargs_632844)
    
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___632846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), chi2_contingency_call_result_632845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_632847 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), getitem___632846, int_632837)
    
    # Assigning a type to the variable 'tuple_var_assignment_632163' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'tuple_var_assignment_632163', subscript_call_result_632847)
    
    # Assigning a Subscript to a Name (line 173):
    
    # Obtaining the type of the subscript
    int_632848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'int')
    
    # Call to chi2_contingency(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'c' (line 173)
    c_632850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 36), 'c', False)
    # Processing the call keyword arguments (line 173)
    str_632851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 47), 'str', 'log-likelihood')
    keyword_632852 = str_632851
    # Getting the type of 'False' (line 173)
    False_632853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 76), 'False', False)
    keyword_632854 = False_632853
    kwargs_632855 = {'correction': keyword_632854, 'lambda_': keyword_632852}
    # Getting the type of 'chi2_contingency' (line 173)
    chi2_contingency_632849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 173)
    chi2_contingency_call_result_632856 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), chi2_contingency_632849, *[c_632850], **kwargs_632855)
    
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___632857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), chi2_contingency_call_result_632856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_632858 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), getitem___632857, int_632848)
    
    # Assigning a type to the variable 'tuple_var_assignment_632164' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'tuple_var_assignment_632164', subscript_call_result_632858)
    
    # Assigning a Subscript to a Name (line 173):
    
    # Obtaining the type of the subscript
    int_632859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'int')
    
    # Call to chi2_contingency(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'c' (line 173)
    c_632861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 36), 'c', False)
    # Processing the call keyword arguments (line 173)
    str_632862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 47), 'str', 'log-likelihood')
    keyword_632863 = str_632862
    # Getting the type of 'False' (line 173)
    False_632864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 76), 'False', False)
    keyword_632865 = False_632864
    kwargs_632866 = {'correction': keyword_632865, 'lambda_': keyword_632863}
    # Getting the type of 'chi2_contingency' (line 173)
    chi2_contingency_632860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 173)
    chi2_contingency_call_result_632867 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), chi2_contingency_632860, *[c_632861], **kwargs_632866)
    
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___632868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), chi2_contingency_call_result_632867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_632869 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), getitem___632868, int_632859)
    
    # Assigning a type to the variable 'tuple_var_assignment_632165' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'tuple_var_assignment_632165', subscript_call_result_632869)
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'tuple_var_assignment_632162' (line 173)
    tuple_var_assignment_632162_632870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'tuple_var_assignment_632162')
    # Assigning a type to the variable 'g' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'g', tuple_var_assignment_632162_632870)
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'tuple_var_assignment_632163' (line 173)
    tuple_var_assignment_632163_632871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'tuple_var_assignment_632163')
    # Assigning a type to the variable 'p' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 7), 'p', tuple_var_assignment_632163_632871)
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'tuple_var_assignment_632164' (line 173)
    tuple_var_assignment_632164_632872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'tuple_var_assignment_632164')
    # Assigning a type to the variable 'dof' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 10), 'dof', tuple_var_assignment_632164_632872)
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'tuple_var_assignment_632165' (line 173)
    tuple_var_assignment_632165_632873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'tuple_var_assignment_632165')
    # Assigning a type to the variable 'e' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'e', tuple_var_assignment_632165_632873)
    
    # Call to assert_allclose(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'g' (line 174)
    g_632875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'g', False)
    int_632876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 23), 'int')
    
    # Call to sum(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_632885 = {}
    
    # Call to xlogy(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'c' (line 174)
    c_632878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'c', False)
    # Getting the type of 'c' (line 174)
    c_632879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 34), 'c', False)
    # Getting the type of 'e' (line 174)
    e_632880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 36), 'e', False)
    # Applying the binary operator 'div' (line 174)
    result_div_632881 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 34), 'div', c_632879, e_632880)
    
    # Processing the call keyword arguments (line 174)
    kwargs_632882 = {}
    # Getting the type of 'xlogy' (line 174)
    xlogy_632877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'xlogy', False)
    # Calling xlogy(args, kwargs) (line 174)
    xlogy_call_result_632883 = invoke(stypy.reporting.localization.Localization(__file__, 174, 25), xlogy_632877, *[c_632878, result_div_632881], **kwargs_632882)
    
    # Obtaining the member 'sum' of a type (line 174)
    sum_632884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 25), xlogy_call_result_632883, 'sum')
    # Calling sum(args, kwargs) (line 174)
    sum_call_result_632886 = invoke(stypy.reporting.localization.Localization(__file__, 174, 25), sum_632884, *[], **kwargs_632885)
    
    # Applying the binary operator '*' (line 174)
    result_mul_632887 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 23), '*', int_632876, sum_call_result_632886)
    
    # Processing the call keyword arguments (line 174)
    kwargs_632888 = {}
    # Getting the type of 'assert_allclose' (line 174)
    assert_allclose_632874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 174)
    assert_allclose_call_result_632889 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), assert_allclose_632874, *[g_632875, result_mul_632887], **kwargs_632888)
    
    
    # Assigning a Call to a Tuple (line 176):
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_632890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'int')
    
    # Call to chi2_contingency(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'c' (line 176)
    c_632892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'c', False)
    # Processing the call keyword arguments (line 176)
    str_632893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'str', 'log-likelihood')
    keyword_632894 = str_632893
    # Getting the type of 'True' (line 176)
    True_632895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 76), 'True', False)
    keyword_632896 = True_632895
    kwargs_632897 = {'correction': keyword_632896, 'lambda_': keyword_632894}
    # Getting the type of 'chi2_contingency' (line 176)
    chi2_contingency_632891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 176)
    chi2_contingency_call_result_632898 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), chi2_contingency_632891, *[c_632892], **kwargs_632897)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___632899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), chi2_contingency_call_result_632898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_632900 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), getitem___632899, int_632890)
    
    # Assigning a type to the variable 'tuple_var_assignment_632166' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_632166', subscript_call_result_632900)
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_632901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'int')
    
    # Call to chi2_contingency(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'c' (line 176)
    c_632903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'c', False)
    # Processing the call keyword arguments (line 176)
    str_632904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'str', 'log-likelihood')
    keyword_632905 = str_632904
    # Getting the type of 'True' (line 176)
    True_632906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 76), 'True', False)
    keyword_632907 = True_632906
    kwargs_632908 = {'correction': keyword_632907, 'lambda_': keyword_632905}
    # Getting the type of 'chi2_contingency' (line 176)
    chi2_contingency_632902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 176)
    chi2_contingency_call_result_632909 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), chi2_contingency_632902, *[c_632903], **kwargs_632908)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___632910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), chi2_contingency_call_result_632909, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_632911 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), getitem___632910, int_632901)
    
    # Assigning a type to the variable 'tuple_var_assignment_632167' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_632167', subscript_call_result_632911)
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_632912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'int')
    
    # Call to chi2_contingency(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'c' (line 176)
    c_632914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'c', False)
    # Processing the call keyword arguments (line 176)
    str_632915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'str', 'log-likelihood')
    keyword_632916 = str_632915
    # Getting the type of 'True' (line 176)
    True_632917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 76), 'True', False)
    keyword_632918 = True_632917
    kwargs_632919 = {'correction': keyword_632918, 'lambda_': keyword_632916}
    # Getting the type of 'chi2_contingency' (line 176)
    chi2_contingency_632913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 176)
    chi2_contingency_call_result_632920 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), chi2_contingency_632913, *[c_632914], **kwargs_632919)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___632921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), chi2_contingency_call_result_632920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_632922 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), getitem___632921, int_632912)
    
    # Assigning a type to the variable 'tuple_var_assignment_632168' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_632168', subscript_call_result_632922)
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_632923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'int')
    
    # Call to chi2_contingency(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'c' (line 176)
    c_632925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'c', False)
    # Processing the call keyword arguments (line 176)
    str_632926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'str', 'log-likelihood')
    keyword_632927 = str_632926
    # Getting the type of 'True' (line 176)
    True_632928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 76), 'True', False)
    keyword_632929 = True_632928
    kwargs_632930 = {'correction': keyword_632929, 'lambda_': keyword_632927}
    # Getting the type of 'chi2_contingency' (line 176)
    chi2_contingency_632924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 176)
    chi2_contingency_call_result_632931 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), chi2_contingency_632924, *[c_632925], **kwargs_632930)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___632932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), chi2_contingency_call_result_632931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_632933 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), getitem___632932, int_632923)
    
    # Assigning a type to the variable 'tuple_var_assignment_632169' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_632169', subscript_call_result_632933)
    
    # Assigning a Name to a Name (line 176):
    # Getting the type of 'tuple_var_assignment_632166' (line 176)
    tuple_var_assignment_632166_632934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_632166')
    # Assigning a type to the variable 'g' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'g', tuple_var_assignment_632166_632934)
    
    # Assigning a Name to a Name (line 176):
    # Getting the type of 'tuple_var_assignment_632167' (line 176)
    tuple_var_assignment_632167_632935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_632167')
    # Assigning a type to the variable 'p' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'p', tuple_var_assignment_632167_632935)
    
    # Assigning a Name to a Name (line 176):
    # Getting the type of 'tuple_var_assignment_632168' (line 176)
    tuple_var_assignment_632168_632936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_632168')
    # Assigning a type to the variable 'dof' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 10), 'dof', tuple_var_assignment_632168_632936)
    
    # Assigning a Name to a Name (line 176):
    # Getting the type of 'tuple_var_assignment_632169' (line 176)
    tuple_var_assignment_632169_632937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_632169')
    # Assigning a type to the variable 'e' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'e', tuple_var_assignment_632169_632937)
    
    # Assigning a BinOp to a Name (line 177):
    
    # Assigning a BinOp to a Name (line 177):
    # Getting the type of 'c' (line 177)
    c_632938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'c')
    
    # Call to array(...): (line 177)
    # Processing the call arguments (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_632941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_632942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    float_632943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 27), list_632942, float_632943)
    # Adding element type (line 177)
    float_632944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 27), list_632942, float_632944)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 26), list_632941, list_632942)
    # Adding element type (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_632945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    float_632946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 41), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 40), list_632945, float_632946)
    # Adding element type (line 177)
    float_632947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 46), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 40), list_632945, float_632947)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 26), list_632941, list_632945)
    
    # Processing the call keyword arguments (line 177)
    kwargs_632948 = {}
    # Getting the type of 'np' (line 177)
    np_632939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'np', False)
    # Obtaining the member 'array' of a type (line 177)
    array_632940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 17), np_632939, 'array')
    # Calling array(args, kwargs) (line 177)
    array_call_result_632949 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), array_632940, *[list_632941], **kwargs_632948)
    
    # Applying the binary operator '+' (line 177)
    result_add_632950 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 13), '+', c_632938, array_call_result_632949)
    
    # Assigning a type to the variable 'c_corr' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'c_corr', result_add_632950)
    
    # Call to assert_allclose(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'g' (line 178)
    g_632952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'g', False)
    int_632953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'int')
    
    # Call to sum(...): (line 178)
    # Processing the call keyword arguments (line 178)
    kwargs_632962 = {}
    
    # Call to xlogy(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'c_corr' (line 178)
    c_corr_632955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 31), 'c_corr', False)
    # Getting the type of 'c_corr' (line 178)
    c_corr_632956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 39), 'c_corr', False)
    # Getting the type of 'e' (line 178)
    e_632957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 46), 'e', False)
    # Applying the binary operator 'div' (line 178)
    result_div_632958 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 39), 'div', c_corr_632956, e_632957)
    
    # Processing the call keyword arguments (line 178)
    kwargs_632959 = {}
    # Getting the type of 'xlogy' (line 178)
    xlogy_632954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'xlogy', False)
    # Calling xlogy(args, kwargs) (line 178)
    xlogy_call_result_632960 = invoke(stypy.reporting.localization.Localization(__file__, 178, 25), xlogy_632954, *[c_corr_632955, result_div_632958], **kwargs_632959)
    
    # Obtaining the member 'sum' of a type (line 178)
    sum_632961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 25), xlogy_call_result_632960, 'sum')
    # Calling sum(args, kwargs) (line 178)
    sum_call_result_632963 = invoke(stypy.reporting.localization.Localization(__file__, 178, 25), sum_632961, *[], **kwargs_632962)
    
    # Applying the binary operator '*' (line 178)
    result_mul_632964 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 23), '*', int_632953, sum_call_result_632963)
    
    # Processing the call keyword arguments (line 178)
    kwargs_632965 = {}
    # Getting the type of 'assert_allclose' (line 178)
    assert_allclose_632951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 178)
    assert_allclose_call_result_632966 = invoke(stypy.reporting.localization.Localization(__file__, 178, 4), assert_allclose_632951, *[g_632952, result_mul_632964], **kwargs_632965)
    
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to array(...): (line 180)
    # Processing the call arguments (line 180)
    
    # Obtaining an instance of the builtin type 'list' (line 180)
    list_632969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 180)
    # Adding element type (line 180)
    
    # Obtaining an instance of the builtin type 'list' (line 180)
    list_632970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 180)
    # Adding element type (line 180)
    int_632971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 18), list_632970, int_632971)
    # Adding element type (line 180)
    int_632972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 18), list_632970, int_632972)
    # Adding element type (line 180)
    int_632973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 18), list_632970, int_632973)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 17), list_632969, list_632970)
    # Adding element type (line 180)
    
    # Obtaining an instance of the builtin type 'list' (line 180)
    list_632974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 180)
    # Adding element type (line 180)
    int_632975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 32), list_632974, int_632975)
    # Adding element type (line 180)
    int_632976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 32), list_632974, int_632976)
    # Adding element type (line 180)
    int_632977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 32), list_632974, int_632977)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 17), list_632969, list_632974)
    
    # Processing the call keyword arguments (line 180)
    kwargs_632978 = {}
    # Getting the type of 'np' (line 180)
    np_632967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 180)
    array_632968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), np_632967, 'array')
    # Calling array(args, kwargs) (line 180)
    array_call_result_632979 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), array_632968, *[list_632969], **kwargs_632978)
    
    # Assigning a type to the variable 'c' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'c', array_call_result_632979)
    
    # Assigning a Call to a Tuple (line 181):
    
    # Assigning a Subscript to a Name (line 181):
    
    # Obtaining the type of the subscript
    int_632980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 4), 'int')
    
    # Call to chi2_contingency(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'c' (line 181)
    c_632982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'c', False)
    # Processing the call keyword arguments (line 181)
    str_632983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 47), 'str', 'log-likelihood')
    keyword_632984 = str_632983
    kwargs_632985 = {'lambda_': keyword_632984}
    # Getting the type of 'chi2_contingency' (line 181)
    chi2_contingency_632981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 181)
    chi2_contingency_call_result_632986 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), chi2_contingency_632981, *[c_632982], **kwargs_632985)
    
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___632987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 4), chi2_contingency_call_result_632986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_632988 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), getitem___632987, int_632980)
    
    # Assigning a type to the variable 'tuple_var_assignment_632170' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_632170', subscript_call_result_632988)
    
    # Assigning a Subscript to a Name (line 181):
    
    # Obtaining the type of the subscript
    int_632989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 4), 'int')
    
    # Call to chi2_contingency(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'c' (line 181)
    c_632991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'c', False)
    # Processing the call keyword arguments (line 181)
    str_632992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 47), 'str', 'log-likelihood')
    keyword_632993 = str_632992
    kwargs_632994 = {'lambda_': keyword_632993}
    # Getting the type of 'chi2_contingency' (line 181)
    chi2_contingency_632990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 181)
    chi2_contingency_call_result_632995 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), chi2_contingency_632990, *[c_632991], **kwargs_632994)
    
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___632996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 4), chi2_contingency_call_result_632995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_632997 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), getitem___632996, int_632989)
    
    # Assigning a type to the variable 'tuple_var_assignment_632171' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_632171', subscript_call_result_632997)
    
    # Assigning a Subscript to a Name (line 181):
    
    # Obtaining the type of the subscript
    int_632998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 4), 'int')
    
    # Call to chi2_contingency(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'c' (line 181)
    c_633000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'c', False)
    # Processing the call keyword arguments (line 181)
    str_633001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 47), 'str', 'log-likelihood')
    keyword_633002 = str_633001
    kwargs_633003 = {'lambda_': keyword_633002}
    # Getting the type of 'chi2_contingency' (line 181)
    chi2_contingency_632999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 181)
    chi2_contingency_call_result_633004 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), chi2_contingency_632999, *[c_633000], **kwargs_633003)
    
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___633005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 4), chi2_contingency_call_result_633004, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_633006 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), getitem___633005, int_632998)
    
    # Assigning a type to the variable 'tuple_var_assignment_632172' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_632172', subscript_call_result_633006)
    
    # Assigning a Subscript to a Name (line 181):
    
    # Obtaining the type of the subscript
    int_633007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 4), 'int')
    
    # Call to chi2_contingency(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'c' (line 181)
    c_633009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'c', False)
    # Processing the call keyword arguments (line 181)
    str_633010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 47), 'str', 'log-likelihood')
    keyword_633011 = str_633010
    kwargs_633012 = {'lambda_': keyword_633011}
    # Getting the type of 'chi2_contingency' (line 181)
    chi2_contingency_633008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'chi2_contingency', False)
    # Calling chi2_contingency(args, kwargs) (line 181)
    chi2_contingency_call_result_633013 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), chi2_contingency_633008, *[c_633009], **kwargs_633012)
    
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___633014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 4), chi2_contingency_call_result_633013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_633015 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), getitem___633014, int_633007)
    
    # Assigning a type to the variable 'tuple_var_assignment_632173' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_632173', subscript_call_result_633015)
    
    # Assigning a Name to a Name (line 181):
    # Getting the type of 'tuple_var_assignment_632170' (line 181)
    tuple_var_assignment_632170_633016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_632170')
    # Assigning a type to the variable 'g' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'g', tuple_var_assignment_632170_633016)
    
    # Assigning a Name to a Name (line 181):
    # Getting the type of 'tuple_var_assignment_632171' (line 181)
    tuple_var_assignment_632171_633017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_632171')
    # Assigning a type to the variable 'p' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 7), 'p', tuple_var_assignment_632171_633017)
    
    # Assigning a Name to a Name (line 181):
    # Getting the type of 'tuple_var_assignment_632172' (line 181)
    tuple_var_assignment_632172_633018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_632172')
    # Assigning a type to the variable 'dof' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 10), 'dof', tuple_var_assignment_632172_633018)
    
    # Assigning a Name to a Name (line 181):
    # Getting the type of 'tuple_var_assignment_632173' (line 181)
    tuple_var_assignment_632173_633019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_632173')
    # Assigning a type to the variable 'e' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'e', tuple_var_assignment_632173_633019)
    
    # Call to assert_allclose(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'g' (line 182)
    g_633021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'g', False)
    int_633022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 23), 'int')
    
    # Call to sum(...): (line 182)
    # Processing the call keyword arguments (line 182)
    kwargs_633031 = {}
    
    # Call to xlogy(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'c' (line 182)
    c_633024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'c', False)
    # Getting the type of 'c' (line 182)
    c_633025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'c', False)
    # Getting the type of 'e' (line 182)
    e_633026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 36), 'e', False)
    # Applying the binary operator 'div' (line 182)
    result_div_633027 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 34), 'div', c_633025, e_633026)
    
    # Processing the call keyword arguments (line 182)
    kwargs_633028 = {}
    # Getting the type of 'xlogy' (line 182)
    xlogy_633023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 25), 'xlogy', False)
    # Calling xlogy(args, kwargs) (line 182)
    xlogy_call_result_633029 = invoke(stypy.reporting.localization.Localization(__file__, 182, 25), xlogy_633023, *[c_633024, result_div_633027], **kwargs_633028)
    
    # Obtaining the member 'sum' of a type (line 182)
    sum_633030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 25), xlogy_call_result_633029, 'sum')
    # Calling sum(args, kwargs) (line 182)
    sum_call_result_633032 = invoke(stypy.reporting.localization.Localization(__file__, 182, 25), sum_633030, *[], **kwargs_633031)
    
    # Applying the binary operator '*' (line 182)
    result_mul_633033 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 23), '*', int_633022, sum_call_result_633032)
    
    # Processing the call keyword arguments (line 182)
    kwargs_633034 = {}
    # Getting the type of 'assert_allclose' (line 182)
    assert_allclose_633020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 182)
    assert_allclose_call_result_633035 = invoke(stypy.reporting.localization.Localization(__file__, 182, 4), assert_allclose_633020, *[g_633021, result_mul_633033], **kwargs_633034)
    
    
    # ################# End of 'test_chi2_contingency_g(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_chi2_contingency_g' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_633036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_633036)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_chi2_contingency_g'
    return stypy_return_type_633036

# Assigning a type to the variable 'test_chi2_contingency_g' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'test_chi2_contingency_g', test_chi2_contingency_g)

@norecursion
def test_chi2_contingency_bad_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_chi2_contingency_bad_args'
    module_type_store = module_type_store.open_function_context('test_chi2_contingency_bad_args', 185, 0, False)
    
    # Passed parameters checking function
    test_chi2_contingency_bad_args.stypy_localization = localization
    test_chi2_contingency_bad_args.stypy_type_of_self = None
    test_chi2_contingency_bad_args.stypy_type_store = module_type_store
    test_chi2_contingency_bad_args.stypy_function_name = 'test_chi2_contingency_bad_args'
    test_chi2_contingency_bad_args.stypy_param_names_list = []
    test_chi2_contingency_bad_args.stypy_varargs_param_name = None
    test_chi2_contingency_bad_args.stypy_kwargs_param_name = None
    test_chi2_contingency_bad_args.stypy_call_defaults = defaults
    test_chi2_contingency_bad_args.stypy_call_varargs = varargs
    test_chi2_contingency_bad_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_chi2_contingency_bad_args', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_chi2_contingency_bad_args', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_chi2_contingency_bad_args(...)' code ##################

    
    # Assigning a Call to a Name (line 189):
    
    # Assigning a Call to a Name (line 189):
    
    # Call to array(...): (line 189)
    # Processing the call arguments (line 189)
    
    # Obtaining an instance of the builtin type 'list' (line 189)
    list_633039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 189)
    # Adding element type (line 189)
    
    # Obtaining an instance of the builtin type 'list' (line 189)
    list_633040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 189)
    # Adding element type (line 189)
    int_633041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 20), list_633040, int_633041)
    # Adding element type (line 189)
    int_633042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 20), list_633040, int_633042)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 19), list_633039, list_633040)
    # Adding element type (line 189)
    
    # Obtaining an instance of the builtin type 'list' (line 189)
    list_633043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 189)
    # Adding element type (line 189)
    int_633044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 30), list_633043, int_633044)
    # Adding element type (line 189)
    int_633045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 30), list_633043, int_633045)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 19), list_633039, list_633043)
    
    # Processing the call keyword arguments (line 189)
    kwargs_633046 = {}
    # Getting the type of 'np' (line 189)
    np_633037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 189)
    array_633038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 10), np_633037, 'array')
    # Calling array(args, kwargs) (line 189)
    array_call_result_633047 = invoke(stypy.reporting.localization.Localization(__file__, 189, 10), array_633038, *[list_633039], **kwargs_633046)
    
    # Assigning a type to the variable 'obs' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'obs', array_call_result_633047)
    
    # Call to assert_raises(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'ValueError' (line 190)
    ValueError_633049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'ValueError', False)
    # Getting the type of 'chi2_contingency' (line 190)
    chi2_contingency_633050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 30), 'chi2_contingency', False)
    # Getting the type of 'obs' (line 190)
    obs_633051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 48), 'obs', False)
    # Processing the call keyword arguments (line 190)
    kwargs_633052 = {}
    # Getting the type of 'assert_raises' (line 190)
    assert_raises_633048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 190)
    assert_raises_call_result_633053 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), assert_raises_633048, *[ValueError_633049, chi2_contingency_633050, obs_633051], **kwargs_633052)
    
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to array(...): (line 194)
    # Processing the call arguments (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_633056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_633057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    int_633058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 20), list_633057, int_633058)
    # Adding element type (line 194)
    int_633059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 20), list_633057, int_633059)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 19), list_633056, list_633057)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_633060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    int_633061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 28), list_633060, int_633061)
    # Adding element type (line 194)
    int_633062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 28), list_633060, int_633062)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 19), list_633056, list_633060)
    
    # Processing the call keyword arguments (line 194)
    kwargs_633063 = {}
    # Getting the type of 'np' (line 194)
    np_633054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 194)
    array_633055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 10), np_633054, 'array')
    # Calling array(args, kwargs) (line 194)
    array_call_result_633064 = invoke(stypy.reporting.localization.Localization(__file__, 194, 10), array_633055, *[list_633056], **kwargs_633063)
    
    # Assigning a type to the variable 'obs' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'obs', array_call_result_633064)
    
    # Call to assert_raises(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'ValueError' (line 195)
    ValueError_633066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'ValueError', False)
    # Getting the type of 'chi2_contingency' (line 195)
    chi2_contingency_633067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 30), 'chi2_contingency', False)
    # Getting the type of 'obs' (line 195)
    obs_633068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 48), 'obs', False)
    # Processing the call keyword arguments (line 195)
    kwargs_633069 = {}
    # Getting the type of 'assert_raises' (line 195)
    assert_raises_633065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 195)
    assert_raises_call_result_633070 = invoke(stypy.reporting.localization.Localization(__file__, 195, 4), assert_raises_633065, *[ValueError_633066, chi2_contingency_633067, obs_633068], **kwargs_633069)
    
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to empty(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining an instance of the builtin type 'tuple' (line 198)
    tuple_633073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 198)
    # Adding element type (line 198)
    int_633074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 20), tuple_633073, int_633074)
    # Adding element type (line 198)
    int_633075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 20), tuple_633073, int_633075)
    
    # Processing the call keyword arguments (line 198)
    kwargs_633076 = {}
    # Getting the type of 'np' (line 198)
    np_633071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 198)
    empty_633072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 10), np_633071, 'empty')
    # Calling empty(args, kwargs) (line 198)
    empty_call_result_633077 = invoke(stypy.reporting.localization.Localization(__file__, 198, 10), empty_633072, *[tuple_633073], **kwargs_633076)
    
    # Assigning a type to the variable 'obs' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'obs', empty_call_result_633077)
    
    # Call to assert_raises(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'ValueError' (line 199)
    ValueError_633079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'ValueError', False)
    # Getting the type of 'chi2_contingency' (line 199)
    chi2_contingency_633080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'chi2_contingency', False)
    # Getting the type of 'obs' (line 199)
    obs_633081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 48), 'obs', False)
    # Processing the call keyword arguments (line 199)
    kwargs_633082 = {}
    # Getting the type of 'assert_raises' (line 199)
    assert_raises_633078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 199)
    assert_raises_call_result_633083 = invoke(stypy.reporting.localization.Localization(__file__, 199, 4), assert_raises_633078, *[ValueError_633079, chi2_contingency_633080, obs_633081], **kwargs_633082)
    
    
    # ################# End of 'test_chi2_contingency_bad_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_chi2_contingency_bad_args' in the type store
    # Getting the type of 'stypy_return_type' (line 185)
    stypy_return_type_633084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_633084)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_chi2_contingency_bad_args'
    return stypy_return_type_633084

# Assigning a type to the variable 'test_chi2_contingency_bad_args' (line 185)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'test_chi2_contingency_bad_args', test_chi2_contingency_bad_args)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
