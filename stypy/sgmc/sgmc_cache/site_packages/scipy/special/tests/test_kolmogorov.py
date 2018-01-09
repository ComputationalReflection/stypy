
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import itertools
4: 
5: import numpy as np
6: from numpy.testing import assert_
7: from scipy.special._testutils import FuncData
8: import pytest
9: 
10: from scipy.special import smirnov, smirnovi, kolmogorov, kolmogi
11: 
12: _rtol = 1e-10
13: 
14: class TestSmirnov(object):
15:     def test_nan(self):
16:         assert_(np.isnan(smirnov(1, np.nan)))
17: 
18:     def test_basic(self):
19:         dataset = [(1, 0.1, 0.9),
20:                    (1, 0.875, 0.125),
21:                    (2, 0.875, 0.125 * 0.125),
22:                    (3, 0.875, 0.125 * 0.125 * 0.125)]
23: 
24:         dataset = np.asarray(dataset)
25:         FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check()
26: 
27:     def test_x_equals_0(self):
28:         dataset = [(n, 0, 1) for n in itertools.chain(range(2, 20), range(1010, 1020))]
29:         dataset = np.asarray(dataset)
30:         FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check()
31: 
32:     def test_x_equals_1(self):
33:         dataset = [(n, 1, 0) for n in itertools.chain(range(2, 20), range(1010, 1020))]
34:         dataset = np.asarray(dataset)
35:         FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check()
36: 
37:     def test_x_equals_0point5(self):
38:         dataset = [(1, 0.5, 0.5),
39:                    (2, 0.5, 0.25),
40:                    (3, 0.5, 0.166666666667),
41:                    (4, 0.5, 0.09375),
42:                    (5, 0.5, 0.056),
43:                    (6, 0.5, 0.0327932098765),
44:                    (7, 0.5, 0.0191958707681),
45:                    (8, 0.5, 0.0112953186035),
46:                    (9, 0.5, 0.00661933257355),
47:                    (10, 0.5, 0.003888705)]
48: 
49:         dataset = np.asarray(dataset)
50:         FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check()
51: 
52:     def test_n_equals_1(self):
53:         x = np.linspace(0, 1, 101, endpoint=True)
54:         dataset = np.column_stack([[1]*len(x), x, 1-x])
55:         # dataset = np.asarray(dataset)
56:         FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check()
57: 
58:     def test_n_equals_2(self):
59:         x = np.linspace(0.5, 1, 101, endpoint=True)
60:         p = np.power(1-x, 2)
61:         n = np.array([2] * len(x))
62:         dataset = np.column_stack([n, x, p])
63:         # dataset = np.asarray(dataset)
64:         FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check()
65: 
66:     def test_n_equals_3(self):
67:         x = np.linspace(0.7, 1, 31, endpoint=True)
68:         p = np.power(1-x, 3)
69:         n = np.array([3] * len(x))
70:         dataset = np.column_stack([n, x, p])
71:         # dataset = np.asarray(dataset)
72:         FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check()
73: 
74:     def test_n_large(self):
75:         # test for large values of n
76:         # Probabilities should go down as n goes up
77:         x = 0.4
78:         pvals = np.array([smirnov(n, x) for n in range(400, 1100, 20)])
79:         dfs = np.diff(pvals)
80:         assert_(np.all(dfs <= 0), msg='Not all diffs negative %s' % dfs)
81: 
82:         dataset = [(1000, 1 - 1.0/2000, np.power(2000.0, -1000))]
83:         dataset = np.asarray(dataset)
84:         FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check()
85: 
86:         # Check asymptotic behaviour
87:         dataset = [(n, 1.0 / np.sqrt(n), np.exp(-2)) for n in range(1000, 5000, 1000)]
88:         dataset = np.asarray(dataset)
89:         FuncData(smirnov, dataset, (0, 1), 2, rtol=.05).check()
90: 
91: 
92: class TestSmirnovi(object):
93:     def test_nan(self):
94:         assert_(np.isnan(smirnovi(1, np.nan)))
95: 
96:     @pytest.mark.xfail(reason="test fails; smirnovi() is not always accurate")
97:     def test_basic(self):
98:         dataset = [(1, 0.4, 0.6),
99:                    (1, 0.6, 0.4),
100:                    (1, 0.99, 0.01),
101:                    (1, 0.01, 0.99),
102:                    (2, 0.125 * 0.125, 0.875),
103:                    (3, 0.125 * 0.125 * 0.125, 0.875),
104:                    (10, 1.0 / 16 ** 10, 1 - 1.0 / 16)]
105: 
106:         dataset = np.asarray(dataset)
107:         FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check()
108: 
109:     @pytest.mark.xfail(reason="test fails; smirnovi(_,0) is not accurate")
110:     def test_x_equals_0(self):
111:         dataset = [(n, 0, 1) for n in itertools.chain(range(2, 20), range(1010, 1020))]
112:         dataset = np.asarray(dataset)
113:         FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check()
114: 
115:     def test_x_equals_1(self):
116:         dataset = [(n, 1, 0) for n in itertools.chain(range(2, 20), range(1010, 1020))]
117:         dataset = np.asarray(dataset)
118:         FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check()
119: 
120:     @pytest.mark.xfail(reason="test fails; smirnovi(1,) is not accurate")
121:     def test_n_equals_1(self):
122:         pp = np.linspace(0, 1, 101, endpoint=True)
123:         dataset = [(1, p, 1-p) for p in pp]
124:         dataset = np.asarray(dataset)
125:         FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check()
126: 
127:     @pytest.mark.xfail(reason="test fails; smirnovi(2,_) is not accurate")
128:     def test_n_equals_2(self):
129:         x = np.linspace(0.5, 1, 101, endpoint=True)
130:         p = np.power(1-x, 2)
131:         n = np.array([2] * len(x))
132:         dataset = np.column_stack([n, p, x])
133:         # dataset = np.asarray(dataset)
134:         FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check()
135: 
136:     @pytest.mark.xfail(reason="test fails; smirnovi(3,_) is not accurate")
137:     def test_n_equals_3(self):
138:         x = np.linspace(0.7, 1, 31, endpoint=True)
139:         p = np.power(1-x, 3)
140:         n = np.array([3] * len(x))
141:         dataset = np.column_stack([n, p, x])
142:         # dataset = np.asarray(dataset)
143:         FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check()
144: 
145:     @pytest.mark.xfail(reason="test fails; smirnovi(_,_) is not accurate")
146:     def test_round_trip(self):
147:         def _sm_smi(n, p):
148:             return smirnov(n, smirnovi(n, p))
149: 
150:         dataset = [(1, 0.4, 0.4),
151:                    (1, 0.6, 0.6),
152:                    (2, 0.875, 0.875),
153:                    (3, 0.875, 0.875),
154:                    (3, 0.125, 0.125),
155:                    (10, 0.999, 0.999),
156:                    (10, 0.0001, 0.0001)]
157: 
158:         dataset = np.asarray(dataset)
159:         FuncData(_sm_smi, dataset, (0, 1), 2, rtol=_rtol).check()
160: 
161:     def test_x_equals_0point5(self):
162:         dataset = [(1, 0.5, 0.5),
163:                    (2, 0.5, 0.366025403784),
164:                    (2, 0.25, 0.5),
165:                    (3, 0.5, 0.297156508177),
166:                    (4, 0.5, 0.255520481121),
167:                    (5, 0.5, 0.234559536069),
168:                    (6, 0.5, 0.21715965898),
169:                    (7, 0.5, 0.202722580034),
170:                    (8, 0.5, 0.190621765256),
171:                    (9, 0.5, 0.180363501362),
172:                    (10, 0.5, 0.17157867006)]
173: 
174:         dataset = np.asarray(dataset)
175:         FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check()
176: 
177: 
178: class TestKolmogorov(object):
179:     def test_nan(self):
180:         assert_(np.isnan(kolmogorov(np.nan)))
181: 
182:     def test_basic(self):
183:         dataset = [(0, 1.0),
184:                    (0.5, 0.96394524366487511),
185:                    (1, 0.26999967167735456),
186:                    (2, 0.00067092525577969533)]
187: 
188:         dataset = np.asarray(dataset)
189:         FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()
190: 
191:     def test_smallx(self):
192:         epsilon = 0.1 ** np.arange(1, 14)
193:         x = np.array([0.571173265106, 0.441027698518, 0.374219690278, 0.331392659217,
194:                       0.300820537459, 0.277539353999, 0.259023494805, 0.243829561254,
195:                       0.231063086389, 0.220135543236, 0.210641372041, 0.202290283658,
196:                       0.19487060742])
197: 
198:         dataset = np.column_stack([x, 1-epsilon])
199:         FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()
200: 
201:     @pytest.mark.xfail(reason="test fails; kolmogi() is not accurate for small p")
202:     def test_round_trip(self):
203:         def _ki_k(_x):
204:             return kolmogi(kolmogorov(_x))
205: 
206:         x = np.linspace(0.0, 2.0, 21, endpoint=True)
207:         dataset = np.column_stack([x, x])
208:         FuncData(_ki_k, dataset, (0,), 1, rtol=_rtol).check()
209: 
210: 
211: class TestKolmogi(object):
212:     def test_nan(self):
213:         assert_(np.isnan(kolmogi(np.nan)))
214: 
215:     @pytest.mark.xfail(reason="test fails; kolmogi() is not accurate for small p")
216:     def test_basic(self):
217:         dataset = [(1.0, 0),
218:                    (0.96394524366487511, 0.5),
219:                    (0.26999967167735456, 1),
220:                    (0.00067092525577969533, 2)]
221: 
222:         dataset = np.asarray(dataset)
223:         FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
224: 
225:     @pytest.mark.xfail(reason="test fails; kolmogi() is not accurate for small p")
226:     def test_smallp(self):
227:         epsilon = 0.1 ** np.arange(1, 14)
228:         x = np.array([0.571173265106, 0.441027698518, 0.374219690278, 0.331392659217,
229:                       0.300820537459, 0.277539353999, 0.259023494805, 0.243829561254,
230:                       0.231063086389, 0.220135543236, 0.210641372041, 0.202290283658,
231:                       0.19487060742])
232: 
233:         dataset = np.column_stack([1-epsilon, x])
234:         FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
235: 
236:     def test_round_trip(self):
237:         def _k_ki(_p):
238:             return kolmogorov(kolmogi(_p))
239: 
240:         p = np.linspace(0.1, 1.0, 10, endpoint=True)
241:         dataset = np.column_stack([p, p])
242:         FuncData(_k_ki, dataset, (0,), 1, rtol=_rtol).check()
243: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import itertools' statement (line 3)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_539730 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_539730) is not StypyTypeError):

    if (import_539730 != 'pyd_module'):
        __import__(import_539730)
        sys_modules_539731 = sys.modules[import_539730]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_539731.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_539730)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_539732 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_539732) is not StypyTypeError):

    if (import_539732 != 'pyd_module'):
        __import__(import_539732)
        sys_modules_539733 = sys.modules[import_539732]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_539733.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_539733, sys_modules_539733.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_539732)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special._testutils import FuncData' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_539734 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._testutils')

if (type(import_539734) is not StypyTypeError):

    if (import_539734 != 'pyd_module'):
        __import__(import_539734)
        sys_modules_539735 = sys.modules[import_539734]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._testutils', sys_modules_539735.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_539735, sys_modules_539735.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._testutils', import_539734)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import pytest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_539736 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_539736) is not StypyTypeError):

    if (import_539736 != 'pyd_module'):
        __import__(import_539736)
        sys_modules_539737 = sys.modules[import_539736]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_539737.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_539736)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.special import smirnov, smirnovi, kolmogorov, kolmogi' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_539738 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special')

if (type(import_539738) is not StypyTypeError):

    if (import_539738 != 'pyd_module'):
        __import__(import_539738)
        sys_modules_539739 = sys.modules[import_539738]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', sys_modules_539739.module_type_store, module_type_store, ['smirnov', 'smirnovi', 'kolmogorov', 'kolmogi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_539739, sys_modules_539739.module_type_store, module_type_store)
    else:
        from scipy.special import smirnov, smirnovi, kolmogorov, kolmogi

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', None, module_type_store, ['smirnov', 'smirnovi', 'kolmogorov', 'kolmogi'], [smirnov, smirnovi, kolmogorov, kolmogi])

else:
    # Assigning a type to the variable 'scipy.special' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', import_539738)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


# Assigning a Num to a Name (line 12):
float_539740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'float')
# Assigning a type to the variable '_rtol' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '_rtol', float_539740)
# Declaration of the 'TestSmirnov' class

class TestSmirnov(object, ):

    @norecursion
    def test_nan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nan'
        module_type_store = module_type_store.open_function_context('test_nan', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_nan')
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_nan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_nan', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nan(...)' code ##################

        
        # Call to assert_(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Call to isnan(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Call to smirnov(...): (line 16)
        # Processing the call arguments (line 16)
        int_539745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'int')
        # Getting the type of 'np' (line 16)
        np_539746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 36), 'np', False)
        # Obtaining the member 'nan' of a type (line 16)
        nan_539747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 36), np_539746, 'nan')
        # Processing the call keyword arguments (line 16)
        kwargs_539748 = {}
        # Getting the type of 'smirnov' (line 16)
        smirnov_539744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'smirnov', False)
        # Calling smirnov(args, kwargs) (line 16)
        smirnov_call_result_539749 = invoke(stypy.reporting.localization.Localization(__file__, 16, 25), smirnov_539744, *[int_539745, nan_539747], **kwargs_539748)
        
        # Processing the call keyword arguments (line 16)
        kwargs_539750 = {}
        # Getting the type of 'np' (line 16)
        np_539742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'np', False)
        # Obtaining the member 'isnan' of a type (line 16)
        isnan_539743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 16), np_539742, 'isnan')
        # Calling isnan(args, kwargs) (line 16)
        isnan_call_result_539751 = invoke(stypy.reporting.localization.Localization(__file__, 16, 16), isnan_539743, *[smirnov_call_result_539749], **kwargs_539750)
        
        # Processing the call keyword arguments (line 16)
        kwargs_539752 = {}
        # Getting the type of 'assert_' (line 16)
        assert__539741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 16)
        assert__call_result_539753 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), assert__539741, *[isnan_call_result_539751], **kwargs_539752)
        
        
        # ################# End of 'test_nan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nan' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_539754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nan'
        return stypy_return_type_539754


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_basic')
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a List to a Name (line 19):
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_539755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 19)
        tuple_539756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        # Adding element type (line 19)
        int_539757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_539756, int_539757)
        # Adding element type (line 19)
        float_539758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_539756, float_539758)
        # Adding element type (line 19)
        float_539759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_539756, float_539759)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 18), list_539755, tuple_539756)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_539760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        int_539761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 20), tuple_539760, int_539761)
        # Adding element type (line 20)
        float_539762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 20), tuple_539760, float_539762)
        # Adding element type (line 20)
        float_539763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 20), tuple_539760, float_539763)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 18), list_539755, tuple_539760)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 21)
        tuple_539764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 21)
        # Adding element type (line 21)
        int_539765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 20), tuple_539764, int_539765)
        # Adding element type (line 21)
        float_539766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 20), tuple_539764, float_539766)
        # Adding element type (line 21)
        float_539767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'float')
        float_539768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 38), 'float')
        # Applying the binary operator '*' (line 21)
        result_mul_539769 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 30), '*', float_539767, float_539768)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 20), tuple_539764, result_mul_539769)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 18), list_539755, tuple_539764)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_539770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        int_539771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), tuple_539770, int_539771)
        # Adding element type (line 22)
        float_539772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), tuple_539770, float_539772)
        # Adding element type (line 22)
        float_539773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'float')
        float_539774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'float')
        # Applying the binary operator '*' (line 22)
        result_mul_539775 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 30), '*', float_539773, float_539774)
        
        float_539776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 46), 'float')
        # Applying the binary operator '*' (line 22)
        result_mul_539777 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 44), '*', result_mul_539775, float_539776)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), tuple_539770, result_mul_539777)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 18), list_539755, tuple_539770)
        
        # Assigning a type to the variable 'dataset' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'dataset', list_539755)
        
        # Assigning a Call to a Name (line 24):
        
        # Call to asarray(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'dataset' (line 24)
        dataset_539780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'dataset', False)
        # Processing the call keyword arguments (line 24)
        kwargs_539781 = {}
        # Getting the type of 'np' (line 24)
        np_539778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 24)
        asarray_539779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 18), np_539778, 'asarray')
        # Calling asarray(args, kwargs) (line 24)
        asarray_call_result_539782 = invoke(stypy.reporting.localization.Localization(__file__, 24, 18), asarray_539779, *[dataset_539780], **kwargs_539781)
        
        # Assigning a type to the variable 'dataset' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'dataset', asarray_call_result_539782)
        
        # Call to check(...): (line 25)
        # Processing the call keyword arguments (line 25)
        kwargs_539795 = {}
        
        # Call to FuncData(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'smirnov' (line 25)
        smirnov_539784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 25)
        dataset_539785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 25)
        tuple_539786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 25)
        # Adding element type (line 25)
        int_539787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), tuple_539786, int_539787)
        # Adding element type (line 25)
        int_539788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), tuple_539786, int_539788)
        
        int_539789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 43), 'int')
        # Processing the call keyword arguments (line 25)
        # Getting the type of '_rtol' (line 25)
        _rtol_539790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 51), '_rtol', False)
        keyword_539791 = _rtol_539790
        kwargs_539792 = {'rtol': keyword_539791}
        # Getting the type of 'FuncData' (line 25)
        FuncData_539783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 25)
        FuncData_call_result_539793 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), FuncData_539783, *[smirnov_539784, dataset_539785, tuple_539786, int_539789], **kwargs_539792)
        
        # Obtaining the member 'check' of a type (line 25)
        check_539794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), FuncData_call_result_539793, 'check')
        # Calling check(args, kwargs) (line 25)
        check_call_result_539796 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), check_539794, *[], **kwargs_539795)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_539797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539797)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_539797


    @norecursion
    def test_x_equals_0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_equals_0'
        module_type_store = module_type_store.open_function_context('test_x_equals_0', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_x_equals_0')
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_x_equals_0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_x_equals_0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_equals_0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_equals_0(...)' code ##################

        
        # Assigning a ListComp to a Name (line 28):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to chain(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Call to range(...): (line 28)
        # Processing the call arguments (line 28)
        int_539805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 60), 'int')
        int_539806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 63), 'int')
        # Processing the call keyword arguments (line 28)
        kwargs_539807 = {}
        # Getting the type of 'range' (line 28)
        range_539804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 54), 'range', False)
        # Calling range(args, kwargs) (line 28)
        range_call_result_539808 = invoke(stypy.reporting.localization.Localization(__file__, 28, 54), range_539804, *[int_539805, int_539806], **kwargs_539807)
        
        
        # Call to range(...): (line 28)
        # Processing the call arguments (line 28)
        int_539810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 74), 'int')
        int_539811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 80), 'int')
        # Processing the call keyword arguments (line 28)
        kwargs_539812 = {}
        # Getting the type of 'range' (line 28)
        range_539809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 68), 'range', False)
        # Calling range(args, kwargs) (line 28)
        range_call_result_539813 = invoke(stypy.reporting.localization.Localization(__file__, 28, 68), range_539809, *[int_539810, int_539811], **kwargs_539812)
        
        # Processing the call keyword arguments (line 28)
        kwargs_539814 = {}
        # Getting the type of 'itertools' (line 28)
        itertools_539802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'itertools', False)
        # Obtaining the member 'chain' of a type (line 28)
        chain_539803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 38), itertools_539802, 'chain')
        # Calling chain(args, kwargs) (line 28)
        chain_call_result_539815 = invoke(stypy.reporting.localization.Localization(__file__, 28, 38), chain_539803, *[range_call_result_539808, range_call_result_539813], **kwargs_539814)
        
        comprehension_539816 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), chain_call_result_539815)
        # Assigning a type to the variable 'n' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'n', comprehension_539816)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_539798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'n' (line 28)
        n_539799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 20), tuple_539798, n_539799)
        # Adding element type (line 28)
        int_539800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 20), tuple_539798, int_539800)
        # Adding element type (line 28)
        int_539801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 20), tuple_539798, int_539801)
        
        list_539817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_539817, tuple_539798)
        # Assigning a type to the variable 'dataset' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'dataset', list_539817)
        
        # Assigning a Call to a Name (line 29):
        
        # Call to asarray(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'dataset' (line 29)
        dataset_539820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'dataset', False)
        # Processing the call keyword arguments (line 29)
        kwargs_539821 = {}
        # Getting the type of 'np' (line 29)
        np_539818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 29)
        asarray_539819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 18), np_539818, 'asarray')
        # Calling asarray(args, kwargs) (line 29)
        asarray_call_result_539822 = invoke(stypy.reporting.localization.Localization(__file__, 29, 18), asarray_539819, *[dataset_539820], **kwargs_539821)
        
        # Assigning a type to the variable 'dataset' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'dataset', asarray_call_result_539822)
        
        # Call to check(...): (line 30)
        # Processing the call keyword arguments (line 30)
        kwargs_539835 = {}
        
        # Call to FuncData(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'smirnov' (line 30)
        smirnov_539824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 30)
        dataset_539825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_539826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        int_539827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), tuple_539826, int_539827)
        # Adding element type (line 30)
        int_539828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), tuple_539826, int_539828)
        
        int_539829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'int')
        # Processing the call keyword arguments (line 30)
        # Getting the type of '_rtol' (line 30)
        _rtol_539830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 51), '_rtol', False)
        keyword_539831 = _rtol_539830
        kwargs_539832 = {'rtol': keyword_539831}
        # Getting the type of 'FuncData' (line 30)
        FuncData_539823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 30)
        FuncData_call_result_539833 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), FuncData_539823, *[smirnov_539824, dataset_539825, tuple_539826, int_539829], **kwargs_539832)
        
        # Obtaining the member 'check' of a type (line 30)
        check_539834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), FuncData_call_result_539833, 'check')
        # Calling check(args, kwargs) (line 30)
        check_call_result_539836 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), check_539834, *[], **kwargs_539835)
        
        
        # ################# End of 'test_x_equals_0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_equals_0' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_539837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_equals_0'
        return stypy_return_type_539837


    @norecursion
    def test_x_equals_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_equals_1'
        module_type_store = module_type_store.open_function_context('test_x_equals_1', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_x_equals_1')
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_x_equals_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_x_equals_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_equals_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_equals_1(...)' code ##################

        
        # Assigning a ListComp to a Name (line 33):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to chain(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to range(...): (line 33)
        # Processing the call arguments (line 33)
        int_539845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 60), 'int')
        int_539846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 63), 'int')
        # Processing the call keyword arguments (line 33)
        kwargs_539847 = {}
        # Getting the type of 'range' (line 33)
        range_539844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 54), 'range', False)
        # Calling range(args, kwargs) (line 33)
        range_call_result_539848 = invoke(stypy.reporting.localization.Localization(__file__, 33, 54), range_539844, *[int_539845, int_539846], **kwargs_539847)
        
        
        # Call to range(...): (line 33)
        # Processing the call arguments (line 33)
        int_539850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 74), 'int')
        int_539851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 80), 'int')
        # Processing the call keyword arguments (line 33)
        kwargs_539852 = {}
        # Getting the type of 'range' (line 33)
        range_539849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 68), 'range', False)
        # Calling range(args, kwargs) (line 33)
        range_call_result_539853 = invoke(stypy.reporting.localization.Localization(__file__, 33, 68), range_539849, *[int_539850, int_539851], **kwargs_539852)
        
        # Processing the call keyword arguments (line 33)
        kwargs_539854 = {}
        # Getting the type of 'itertools' (line 33)
        itertools_539842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'itertools', False)
        # Obtaining the member 'chain' of a type (line 33)
        chain_539843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 38), itertools_539842, 'chain')
        # Calling chain(args, kwargs) (line 33)
        chain_call_result_539855 = invoke(stypy.reporting.localization.Localization(__file__, 33, 38), chain_539843, *[range_call_result_539848, range_call_result_539853], **kwargs_539854)
        
        comprehension_539856 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), chain_call_result_539855)
        # Assigning a type to the variable 'n' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'n', comprehension_539856)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_539838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        # Getting the type of 'n' (line 33)
        n_539839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), tuple_539838, n_539839)
        # Adding element type (line 33)
        int_539840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), tuple_539838, int_539840)
        # Adding element type (line 33)
        int_539841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), tuple_539838, int_539841)
        
        list_539857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), list_539857, tuple_539838)
        # Assigning a type to the variable 'dataset' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'dataset', list_539857)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to asarray(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'dataset' (line 34)
        dataset_539860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 29), 'dataset', False)
        # Processing the call keyword arguments (line 34)
        kwargs_539861 = {}
        # Getting the type of 'np' (line 34)
        np_539858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 34)
        asarray_539859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 18), np_539858, 'asarray')
        # Calling asarray(args, kwargs) (line 34)
        asarray_call_result_539862 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), asarray_539859, *[dataset_539860], **kwargs_539861)
        
        # Assigning a type to the variable 'dataset' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'dataset', asarray_call_result_539862)
        
        # Call to check(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_539875 = {}
        
        # Call to FuncData(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'smirnov' (line 35)
        smirnov_539864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 35)
        dataset_539865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_539866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        int_539867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 36), tuple_539866, int_539867)
        # Adding element type (line 35)
        int_539868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 36), tuple_539866, int_539868)
        
        int_539869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'int')
        # Processing the call keyword arguments (line 35)
        # Getting the type of '_rtol' (line 35)
        _rtol_539870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 51), '_rtol', False)
        keyword_539871 = _rtol_539870
        kwargs_539872 = {'rtol': keyword_539871}
        # Getting the type of 'FuncData' (line 35)
        FuncData_539863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 35)
        FuncData_call_result_539873 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), FuncData_539863, *[smirnov_539864, dataset_539865, tuple_539866, int_539869], **kwargs_539872)
        
        # Obtaining the member 'check' of a type (line 35)
        check_539874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), FuncData_call_result_539873, 'check')
        # Calling check(args, kwargs) (line 35)
        check_call_result_539876 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), check_539874, *[], **kwargs_539875)
        
        
        # ################# End of 'test_x_equals_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_equals_1' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_539877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_equals_1'
        return stypy_return_type_539877


    @norecursion
    def test_x_equals_0point5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_equals_0point5'
        module_type_store = module_type_store.open_function_context('test_x_equals_0point5', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_x_equals_0point5')
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_x_equals_0point5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_x_equals_0point5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_equals_0point5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_equals_0point5(...)' code ##################

        
        # Assigning a List to a Name (line 38):
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_539878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_539879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        int_539880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), tuple_539879, int_539880)
        # Adding element type (line 38)
        float_539881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), tuple_539879, float_539881)
        # Adding element type (line 38)
        float_539882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), tuple_539879, float_539882)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539879)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_539883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_539884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), tuple_539883, int_539884)
        # Adding element type (line 39)
        float_539885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), tuple_539883, float_539885)
        # Adding element type (line 39)
        float_539886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), tuple_539883, float_539886)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539883)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_539887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        int_539888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), tuple_539887, int_539888)
        # Adding element type (line 40)
        float_539889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), tuple_539887, float_539889)
        # Adding element type (line 40)
        float_539890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), tuple_539887, float_539890)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539887)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_539891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        int_539892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), tuple_539891, int_539892)
        # Adding element type (line 41)
        float_539893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), tuple_539891, float_539893)
        # Adding element type (line 41)
        float_539894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), tuple_539891, float_539894)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539891)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_539895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        int_539896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 20), tuple_539895, int_539896)
        # Adding element type (line 42)
        float_539897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 20), tuple_539895, float_539897)
        # Adding element type (line 42)
        float_539898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 20), tuple_539895, float_539898)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539895)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 43)
        tuple_539899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 43)
        # Adding element type (line 43)
        int_539900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), tuple_539899, int_539900)
        # Adding element type (line 43)
        float_539901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), tuple_539899, float_539901)
        # Adding element type (line 43)
        float_539902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), tuple_539899, float_539902)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539899)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 44)
        tuple_539903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 44)
        # Adding element type (line 44)
        int_539904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 20), tuple_539903, int_539904)
        # Adding element type (line 44)
        float_539905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 20), tuple_539903, float_539905)
        # Adding element type (line 44)
        float_539906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 20), tuple_539903, float_539906)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539903)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 45)
        tuple_539907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 45)
        # Adding element type (line 45)
        int_539908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 20), tuple_539907, int_539908)
        # Adding element type (line 45)
        float_539909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 20), tuple_539907, float_539909)
        # Adding element type (line 45)
        float_539910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 20), tuple_539907, float_539910)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539907)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_539911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        int_539912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 20), tuple_539911, int_539912)
        # Adding element type (line 46)
        float_539913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 20), tuple_539911, float_539913)
        # Adding element type (line 46)
        float_539914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 20), tuple_539911, float_539914)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539911)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_539915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        int_539916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 20), tuple_539915, int_539916)
        # Adding element type (line 47)
        float_539917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 20), tuple_539915, float_539917)
        # Adding element type (line 47)
        float_539918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 20), tuple_539915, float_539918)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 18), list_539878, tuple_539915)
        
        # Assigning a type to the variable 'dataset' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'dataset', list_539878)
        
        # Assigning a Call to a Name (line 49):
        
        # Call to asarray(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'dataset' (line 49)
        dataset_539921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'dataset', False)
        # Processing the call keyword arguments (line 49)
        kwargs_539922 = {}
        # Getting the type of 'np' (line 49)
        np_539919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 49)
        asarray_539920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 18), np_539919, 'asarray')
        # Calling asarray(args, kwargs) (line 49)
        asarray_call_result_539923 = invoke(stypy.reporting.localization.Localization(__file__, 49, 18), asarray_539920, *[dataset_539921], **kwargs_539922)
        
        # Assigning a type to the variable 'dataset' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'dataset', asarray_call_result_539923)
        
        # Call to check(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_539936 = {}
        
        # Call to FuncData(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'smirnov' (line 50)
        smirnov_539925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 50)
        dataset_539926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_539927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        int_539928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 36), tuple_539927, int_539928)
        # Adding element type (line 50)
        int_539929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 36), tuple_539927, int_539929)
        
        int_539930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 43), 'int')
        # Processing the call keyword arguments (line 50)
        # Getting the type of '_rtol' (line 50)
        _rtol_539931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 51), '_rtol', False)
        keyword_539932 = _rtol_539931
        kwargs_539933 = {'rtol': keyword_539932}
        # Getting the type of 'FuncData' (line 50)
        FuncData_539924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 50)
        FuncData_call_result_539934 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), FuncData_539924, *[smirnov_539925, dataset_539926, tuple_539927, int_539930], **kwargs_539933)
        
        # Obtaining the member 'check' of a type (line 50)
        check_539935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), FuncData_call_result_539934, 'check')
        # Calling check(args, kwargs) (line 50)
        check_call_result_539937 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), check_539935, *[], **kwargs_539936)
        
        
        # ################# End of 'test_x_equals_0point5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_equals_0point5' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_539938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_equals_0point5'
        return stypy_return_type_539938


    @norecursion
    def test_n_equals_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_n_equals_1'
        module_type_store = module_type_store.open_function_context('test_n_equals_1', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_n_equals_1')
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_n_equals_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_n_equals_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_n_equals_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_n_equals_1(...)' code ##################

        
        # Assigning a Call to a Name (line 53):
        
        # Call to linspace(...): (line 53)
        # Processing the call arguments (line 53)
        int_539941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'int')
        int_539942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'int')
        int_539943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 30), 'int')
        # Processing the call keyword arguments (line 53)
        # Getting the type of 'True' (line 53)
        True_539944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 44), 'True', False)
        keyword_539945 = True_539944
        kwargs_539946 = {'endpoint': keyword_539945}
        # Getting the type of 'np' (line 53)
        np_539939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 53)
        linspace_539940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), np_539939, 'linspace')
        # Calling linspace(args, kwargs) (line 53)
        linspace_call_result_539947 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), linspace_539940, *[int_539941, int_539942, int_539943], **kwargs_539946)
        
        # Assigning a type to the variable 'x' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'x', linspace_call_result_539947)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to column_stack(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_539950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_539951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_539952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 35), list_539951, int_539952)
        
        
        # Call to len(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'x' (line 54)
        x_539954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 43), 'x', False)
        # Processing the call keyword arguments (line 54)
        kwargs_539955 = {}
        # Getting the type of 'len' (line 54)
        len_539953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'len', False)
        # Calling len(args, kwargs) (line 54)
        len_call_result_539956 = invoke(stypy.reporting.localization.Localization(__file__, 54, 39), len_539953, *[x_539954], **kwargs_539955)
        
        # Applying the binary operator '*' (line 54)
        result_mul_539957 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 35), '*', list_539951, len_call_result_539956)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), list_539950, result_mul_539957)
        # Adding element type (line 54)
        # Getting the type of 'x' (line 54)
        x_539958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 47), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), list_539950, x_539958)
        # Adding element type (line 54)
        int_539959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 50), 'int')
        # Getting the type of 'x' (line 54)
        x_539960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 52), 'x', False)
        # Applying the binary operator '-' (line 54)
        result_sub_539961 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 50), '-', int_539959, x_539960)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), list_539950, result_sub_539961)
        
        # Processing the call keyword arguments (line 54)
        kwargs_539962 = {}
        # Getting the type of 'np' (line 54)
        np_539948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 54)
        column_stack_539949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 18), np_539948, 'column_stack')
        # Calling column_stack(args, kwargs) (line 54)
        column_stack_call_result_539963 = invoke(stypy.reporting.localization.Localization(__file__, 54, 18), column_stack_539949, *[list_539950], **kwargs_539962)
        
        # Assigning a type to the variable 'dataset' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'dataset', column_stack_call_result_539963)
        
        # Call to check(...): (line 56)
        # Processing the call keyword arguments (line 56)
        kwargs_539976 = {}
        
        # Call to FuncData(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'smirnov' (line 56)
        smirnov_539965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 56)
        dataset_539966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_539967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        int_539968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 36), tuple_539967, int_539968)
        # Adding element type (line 56)
        int_539969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 36), tuple_539967, int_539969)
        
        int_539970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 43), 'int')
        # Processing the call keyword arguments (line 56)
        # Getting the type of '_rtol' (line 56)
        _rtol_539971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 51), '_rtol', False)
        keyword_539972 = _rtol_539971
        kwargs_539973 = {'rtol': keyword_539972}
        # Getting the type of 'FuncData' (line 56)
        FuncData_539964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 56)
        FuncData_call_result_539974 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), FuncData_539964, *[smirnov_539965, dataset_539966, tuple_539967, int_539970], **kwargs_539973)
        
        # Obtaining the member 'check' of a type (line 56)
        check_539975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), FuncData_call_result_539974, 'check')
        # Calling check(args, kwargs) (line 56)
        check_call_result_539977 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), check_539975, *[], **kwargs_539976)
        
        
        # ################# End of 'test_n_equals_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_n_equals_1' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_539978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539978)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_n_equals_1'
        return stypy_return_type_539978


    @norecursion
    def test_n_equals_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_n_equals_2'
        module_type_store = module_type_store.open_function_context('test_n_equals_2', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_n_equals_2')
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_n_equals_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_n_equals_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_n_equals_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_n_equals_2(...)' code ##################

        
        # Assigning a Call to a Name (line 59):
        
        # Call to linspace(...): (line 59)
        # Processing the call arguments (line 59)
        float_539981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'float')
        int_539982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 29), 'int')
        int_539983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 32), 'int')
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'True' (line 59)
        True_539984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 46), 'True', False)
        keyword_539985 = True_539984
        kwargs_539986 = {'endpoint': keyword_539985}
        # Getting the type of 'np' (line 59)
        np_539979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 59)
        linspace_539980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), np_539979, 'linspace')
        # Calling linspace(args, kwargs) (line 59)
        linspace_call_result_539987 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), linspace_539980, *[float_539981, int_539982, int_539983], **kwargs_539986)
        
        # Assigning a type to the variable 'x' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'x', linspace_call_result_539987)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to power(...): (line 60)
        # Processing the call arguments (line 60)
        int_539990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 21), 'int')
        # Getting the type of 'x' (line 60)
        x_539991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'x', False)
        # Applying the binary operator '-' (line 60)
        result_sub_539992 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 21), '-', int_539990, x_539991)
        
        int_539993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 26), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_539994 = {}
        # Getting the type of 'np' (line 60)
        np_539988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'np', False)
        # Obtaining the member 'power' of a type (line 60)
        power_539989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), np_539988, 'power')
        # Calling power(args, kwargs) (line 60)
        power_call_result_539995 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), power_539989, *[result_sub_539992, int_539993], **kwargs_539994)
        
        # Assigning a type to the variable 'p' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'p', power_call_result_539995)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to array(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_539998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_539999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 21), list_539998, int_539999)
        
        
        # Call to len(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'x' (line 61)
        x_540001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'x', False)
        # Processing the call keyword arguments (line 61)
        kwargs_540002 = {}
        # Getting the type of 'len' (line 61)
        len_540000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), 'len', False)
        # Calling len(args, kwargs) (line 61)
        len_call_result_540003 = invoke(stypy.reporting.localization.Localization(__file__, 61, 27), len_540000, *[x_540001], **kwargs_540002)
        
        # Applying the binary operator '*' (line 61)
        result_mul_540004 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 21), '*', list_539998, len_call_result_540003)
        
        # Processing the call keyword arguments (line 61)
        kwargs_540005 = {}
        # Getting the type of 'np' (line 61)
        np_539996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 61)
        array_539997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), np_539996, 'array')
        # Calling array(args, kwargs) (line 61)
        array_call_result_540006 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), array_539997, *[result_mul_540004], **kwargs_540005)
        
        # Assigning a type to the variable 'n' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'n', array_call_result_540006)
        
        # Assigning a Call to a Name (line 62):
        
        # Call to column_stack(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_540009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        # Getting the type of 'n' (line 62)
        n_540010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 34), list_540009, n_540010)
        # Adding element type (line 62)
        # Getting the type of 'x' (line 62)
        x_540011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 34), list_540009, x_540011)
        # Adding element type (line 62)
        # Getting the type of 'p' (line 62)
        p_540012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 41), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 34), list_540009, p_540012)
        
        # Processing the call keyword arguments (line 62)
        kwargs_540013 = {}
        # Getting the type of 'np' (line 62)
        np_540007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 62)
        column_stack_540008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 18), np_540007, 'column_stack')
        # Calling column_stack(args, kwargs) (line 62)
        column_stack_call_result_540014 = invoke(stypy.reporting.localization.Localization(__file__, 62, 18), column_stack_540008, *[list_540009], **kwargs_540013)
        
        # Assigning a type to the variable 'dataset' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'dataset', column_stack_call_result_540014)
        
        # Call to check(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_540027 = {}
        
        # Call to FuncData(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'smirnov' (line 64)
        smirnov_540016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 64)
        dataset_540017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 64)
        tuple_540018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 64)
        # Adding element type (line 64)
        int_540019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 36), tuple_540018, int_540019)
        # Adding element type (line 64)
        int_540020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 36), tuple_540018, int_540020)
        
        int_540021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 43), 'int')
        # Processing the call keyword arguments (line 64)
        # Getting the type of '_rtol' (line 64)
        _rtol_540022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 51), '_rtol', False)
        keyword_540023 = _rtol_540022
        kwargs_540024 = {'rtol': keyword_540023}
        # Getting the type of 'FuncData' (line 64)
        FuncData_540015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 64)
        FuncData_call_result_540025 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), FuncData_540015, *[smirnov_540016, dataset_540017, tuple_540018, int_540021], **kwargs_540024)
        
        # Obtaining the member 'check' of a type (line 64)
        check_540026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), FuncData_call_result_540025, 'check')
        # Calling check(args, kwargs) (line 64)
        check_call_result_540028 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), check_540026, *[], **kwargs_540027)
        
        
        # ################# End of 'test_n_equals_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_n_equals_2' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_540029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_n_equals_2'
        return stypy_return_type_540029


    @norecursion
    def test_n_equals_3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_n_equals_3'
        module_type_store = module_type_store.open_function_context('test_n_equals_3', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_n_equals_3')
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_n_equals_3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_n_equals_3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_n_equals_3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_n_equals_3(...)' code ##################

        
        # Assigning a Call to a Name (line 67):
        
        # Call to linspace(...): (line 67)
        # Processing the call arguments (line 67)
        float_540032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 24), 'float')
        int_540033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'int')
        int_540034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 32), 'int')
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'True' (line 67)
        True_540035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 45), 'True', False)
        keyword_540036 = True_540035
        kwargs_540037 = {'endpoint': keyword_540036}
        # Getting the type of 'np' (line 67)
        np_540030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 67)
        linspace_540031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), np_540030, 'linspace')
        # Calling linspace(args, kwargs) (line 67)
        linspace_call_result_540038 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), linspace_540031, *[float_540032, int_540033, int_540034], **kwargs_540037)
        
        # Assigning a type to the variable 'x' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'x', linspace_call_result_540038)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to power(...): (line 68)
        # Processing the call arguments (line 68)
        int_540041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'int')
        # Getting the type of 'x' (line 68)
        x_540042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'x', False)
        # Applying the binary operator '-' (line 68)
        result_sub_540043 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 21), '-', int_540041, x_540042)
        
        int_540044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'int')
        # Processing the call keyword arguments (line 68)
        kwargs_540045 = {}
        # Getting the type of 'np' (line 68)
        np_540039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'np', False)
        # Obtaining the member 'power' of a type (line 68)
        power_540040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), np_540039, 'power')
        # Calling power(args, kwargs) (line 68)
        power_call_result_540046 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), power_540040, *[result_sub_540043, int_540044], **kwargs_540045)
        
        # Assigning a type to the variable 'p' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'p', power_call_result_540046)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to array(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_540049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        int_540050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 21), list_540049, int_540050)
        
        
        # Call to len(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'x' (line 69)
        x_540052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'x', False)
        # Processing the call keyword arguments (line 69)
        kwargs_540053 = {}
        # Getting the type of 'len' (line 69)
        len_540051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'len', False)
        # Calling len(args, kwargs) (line 69)
        len_call_result_540054 = invoke(stypy.reporting.localization.Localization(__file__, 69, 27), len_540051, *[x_540052], **kwargs_540053)
        
        # Applying the binary operator '*' (line 69)
        result_mul_540055 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 21), '*', list_540049, len_call_result_540054)
        
        # Processing the call keyword arguments (line 69)
        kwargs_540056 = {}
        # Getting the type of 'np' (line 69)
        np_540047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 69)
        array_540048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), np_540047, 'array')
        # Calling array(args, kwargs) (line 69)
        array_call_result_540057 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), array_540048, *[result_mul_540055], **kwargs_540056)
        
        # Assigning a type to the variable 'n' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'n', array_call_result_540057)
        
        # Assigning a Call to a Name (line 70):
        
        # Call to column_stack(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_540060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        # Getting the type of 'n' (line 70)
        n_540061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 34), list_540060, n_540061)
        # Adding element type (line 70)
        # Getting the type of 'x' (line 70)
        x_540062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 34), list_540060, x_540062)
        # Adding element type (line 70)
        # Getting the type of 'p' (line 70)
        p_540063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 41), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 34), list_540060, p_540063)
        
        # Processing the call keyword arguments (line 70)
        kwargs_540064 = {}
        # Getting the type of 'np' (line 70)
        np_540058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 70)
        column_stack_540059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 18), np_540058, 'column_stack')
        # Calling column_stack(args, kwargs) (line 70)
        column_stack_call_result_540065 = invoke(stypy.reporting.localization.Localization(__file__, 70, 18), column_stack_540059, *[list_540060], **kwargs_540064)
        
        # Assigning a type to the variable 'dataset' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'dataset', column_stack_call_result_540065)
        
        # Call to check(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_540078 = {}
        
        # Call to FuncData(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'smirnov' (line 72)
        smirnov_540067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 72)
        dataset_540068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_540069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        int_540070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 36), tuple_540069, int_540070)
        # Adding element type (line 72)
        int_540071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 36), tuple_540069, int_540071)
        
        int_540072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 43), 'int')
        # Processing the call keyword arguments (line 72)
        # Getting the type of '_rtol' (line 72)
        _rtol_540073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 51), '_rtol', False)
        keyword_540074 = _rtol_540073
        kwargs_540075 = {'rtol': keyword_540074}
        # Getting the type of 'FuncData' (line 72)
        FuncData_540066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 72)
        FuncData_call_result_540076 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), FuncData_540066, *[smirnov_540067, dataset_540068, tuple_540069, int_540072], **kwargs_540075)
        
        # Obtaining the member 'check' of a type (line 72)
        check_540077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), FuncData_call_result_540076, 'check')
        # Calling check(args, kwargs) (line 72)
        check_call_result_540079 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), check_540077, *[], **kwargs_540078)
        
        
        # ################# End of 'test_n_equals_3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_n_equals_3' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_540080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540080)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_n_equals_3'
        return stypy_return_type_540080


    @norecursion
    def test_n_large(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_n_large'
        module_type_store = module_type_store.open_function_context('test_n_large', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_function_name', 'TestSmirnov.test_n_large')
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnov.test_n_large.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.test_n_large', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_n_large', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_n_large(...)' code ##################

        
        # Assigning a Num to a Name (line 77):
        float_540081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 12), 'float')
        # Assigning a type to the variable 'x' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'x', float_540081)
        
        # Assigning a Call to a Name (line 78):
        
        # Call to array(...): (line 78)
        # Processing the call arguments (line 78)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 78)
        # Processing the call arguments (line 78)
        int_540090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 55), 'int')
        int_540091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 60), 'int')
        int_540092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 66), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_540093 = {}
        # Getting the type of 'range' (line 78)
        range_540089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 49), 'range', False)
        # Calling range(args, kwargs) (line 78)
        range_call_result_540094 = invoke(stypy.reporting.localization.Localization(__file__, 78, 49), range_540089, *[int_540090, int_540091, int_540092], **kwargs_540093)
        
        comprehension_540095 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), range_call_result_540094)
        # Assigning a type to the variable 'n' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'n', comprehension_540095)
        
        # Call to smirnov(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'n' (line 78)
        n_540085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'n', False)
        # Getting the type of 'x' (line 78)
        x_540086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'x', False)
        # Processing the call keyword arguments (line 78)
        kwargs_540087 = {}
        # Getting the type of 'smirnov' (line 78)
        smirnov_540084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'smirnov', False)
        # Calling smirnov(args, kwargs) (line 78)
        smirnov_call_result_540088 = invoke(stypy.reporting.localization.Localization(__file__, 78, 26), smirnov_540084, *[n_540085, x_540086], **kwargs_540087)
        
        list_540096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), list_540096, smirnov_call_result_540088)
        # Processing the call keyword arguments (line 78)
        kwargs_540097 = {}
        # Getting the type of 'np' (line 78)
        np_540082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 78)
        array_540083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), np_540082, 'array')
        # Calling array(args, kwargs) (line 78)
        array_call_result_540098 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), array_540083, *[list_540096], **kwargs_540097)
        
        # Assigning a type to the variable 'pvals' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'pvals', array_call_result_540098)
        
        # Assigning a Call to a Name (line 79):
        
        # Call to diff(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'pvals' (line 79)
        pvals_540101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'pvals', False)
        # Processing the call keyword arguments (line 79)
        kwargs_540102 = {}
        # Getting the type of 'np' (line 79)
        np_540099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 14), 'np', False)
        # Obtaining the member 'diff' of a type (line 79)
        diff_540100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 14), np_540099, 'diff')
        # Calling diff(args, kwargs) (line 79)
        diff_call_result_540103 = invoke(stypy.reporting.localization.Localization(__file__, 79, 14), diff_540100, *[pvals_540101], **kwargs_540102)
        
        # Assigning a type to the variable 'dfs' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'dfs', diff_call_result_540103)
        
        # Call to assert_(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to all(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Getting the type of 'dfs' (line 80)
        dfs_540107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'dfs', False)
        int_540108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 30), 'int')
        # Applying the binary operator '<=' (line 80)
        result_le_540109 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 23), '<=', dfs_540107, int_540108)
        
        # Processing the call keyword arguments (line 80)
        kwargs_540110 = {}
        # Getting the type of 'np' (line 80)
        np_540105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 80)
        all_540106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), np_540105, 'all')
        # Calling all(args, kwargs) (line 80)
        all_call_result_540111 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), all_540106, *[result_le_540109], **kwargs_540110)
        
        # Processing the call keyword arguments (line 80)
        str_540112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 38), 'str', 'Not all diffs negative %s')
        # Getting the type of 'dfs' (line 80)
        dfs_540113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 68), 'dfs', False)
        # Applying the binary operator '%' (line 80)
        result_mod_540114 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 38), '%', str_540112, dfs_540113)
        
        keyword_540115 = result_mod_540114
        kwargs_540116 = {'msg': keyword_540115}
        # Getting the type of 'assert_' (line 80)
        assert__540104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 80)
        assert__call_result_540117 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert__540104, *[all_call_result_540111], **kwargs_540116)
        
        
        # Assigning a List to a Name (line 82):
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_540118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_540119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        int_540120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), tuple_540119, int_540120)
        # Adding element type (line 82)
        int_540121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'int')
        float_540122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'float')
        int_540123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 34), 'int')
        # Applying the binary operator 'div' (line 82)
        result_div_540124 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 30), 'div', float_540122, int_540123)
        
        # Applying the binary operator '-' (line 82)
        result_sub_540125 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 26), '-', int_540121, result_div_540124)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), tuple_540119, result_sub_540125)
        # Adding element type (line 82)
        
        # Call to power(...): (line 82)
        # Processing the call arguments (line 82)
        float_540128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 49), 'float')
        int_540129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 57), 'int')
        # Processing the call keyword arguments (line 82)
        kwargs_540130 = {}
        # Getting the type of 'np' (line 82)
        np_540126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'np', False)
        # Obtaining the member 'power' of a type (line 82)
        power_540127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), np_540126, 'power')
        # Calling power(args, kwargs) (line 82)
        power_call_result_540131 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), power_540127, *[float_540128, int_540129], **kwargs_540130)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), tuple_540119, power_call_result_540131)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), list_540118, tuple_540119)
        
        # Assigning a type to the variable 'dataset' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'dataset', list_540118)
        
        # Assigning a Call to a Name (line 83):
        
        # Call to asarray(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'dataset' (line 83)
        dataset_540134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'dataset', False)
        # Processing the call keyword arguments (line 83)
        kwargs_540135 = {}
        # Getting the type of 'np' (line 83)
        np_540132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 83)
        asarray_540133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 18), np_540132, 'asarray')
        # Calling asarray(args, kwargs) (line 83)
        asarray_call_result_540136 = invoke(stypy.reporting.localization.Localization(__file__, 83, 18), asarray_540133, *[dataset_540134], **kwargs_540135)
        
        # Assigning a type to the variable 'dataset' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'dataset', asarray_call_result_540136)
        
        # Call to check(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_540149 = {}
        
        # Call to FuncData(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'smirnov' (line 84)
        smirnov_540138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 84)
        dataset_540139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_540140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        int_540141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 36), tuple_540140, int_540141)
        # Adding element type (line 84)
        int_540142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 36), tuple_540140, int_540142)
        
        int_540143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 43), 'int')
        # Processing the call keyword arguments (line 84)
        # Getting the type of '_rtol' (line 84)
        _rtol_540144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), '_rtol', False)
        keyword_540145 = _rtol_540144
        kwargs_540146 = {'rtol': keyword_540145}
        # Getting the type of 'FuncData' (line 84)
        FuncData_540137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 84)
        FuncData_call_result_540147 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), FuncData_540137, *[smirnov_540138, dataset_540139, tuple_540140, int_540143], **kwargs_540146)
        
        # Obtaining the member 'check' of a type (line 84)
        check_540148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), FuncData_call_result_540147, 'check')
        # Calling check(args, kwargs) (line 84)
        check_call_result_540150 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), check_540148, *[], **kwargs_540149)
        
        
        # Assigning a ListComp to a Name (line 87):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 87)
        # Processing the call arguments (line 87)
        int_540166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 68), 'int')
        int_540167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 74), 'int')
        int_540168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 80), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_540169 = {}
        # Getting the type of 'range' (line 87)
        range_540165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 62), 'range', False)
        # Calling range(args, kwargs) (line 87)
        range_call_result_540170 = invoke(stypy.reporting.localization.Localization(__file__, 87, 62), range_540165, *[int_540166, int_540167, int_540168], **kwargs_540169)
        
        comprehension_540171 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 19), range_call_result_540170)
        # Assigning a type to the variable 'n' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'n', comprehension_540171)
        
        # Obtaining an instance of the builtin type 'tuple' (line 87)
        tuple_540151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 87)
        # Adding element type (line 87)
        # Getting the type of 'n' (line 87)
        n_540152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 20), tuple_540151, n_540152)
        # Adding element type (line 87)
        float_540153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'float')
        
        # Call to sqrt(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'n' (line 87)
        n_540156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'n', False)
        # Processing the call keyword arguments (line 87)
        kwargs_540157 = {}
        # Getting the type of 'np' (line 87)
        np_540154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 87)
        sqrt_540155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 29), np_540154, 'sqrt')
        # Calling sqrt(args, kwargs) (line 87)
        sqrt_call_result_540158 = invoke(stypy.reporting.localization.Localization(__file__, 87, 29), sqrt_540155, *[n_540156], **kwargs_540157)
        
        # Applying the binary operator 'div' (line 87)
        result_div_540159 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 23), 'div', float_540153, sqrt_call_result_540158)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 20), tuple_540151, result_div_540159)
        # Adding element type (line 87)
        
        # Call to exp(...): (line 87)
        # Processing the call arguments (line 87)
        int_540162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 48), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_540163 = {}
        # Getting the type of 'np' (line 87)
        np_540160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'np', False)
        # Obtaining the member 'exp' of a type (line 87)
        exp_540161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 41), np_540160, 'exp')
        # Calling exp(args, kwargs) (line 87)
        exp_call_result_540164 = invoke(stypy.reporting.localization.Localization(__file__, 87, 41), exp_540161, *[int_540162], **kwargs_540163)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 20), tuple_540151, exp_call_result_540164)
        
        list_540172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 19), list_540172, tuple_540151)
        # Assigning a type to the variable 'dataset' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'dataset', list_540172)
        
        # Assigning a Call to a Name (line 88):
        
        # Call to asarray(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'dataset' (line 88)
        dataset_540175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'dataset', False)
        # Processing the call keyword arguments (line 88)
        kwargs_540176 = {}
        # Getting the type of 'np' (line 88)
        np_540173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 88)
        asarray_540174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 18), np_540173, 'asarray')
        # Calling asarray(args, kwargs) (line 88)
        asarray_call_result_540177 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), asarray_540174, *[dataset_540175], **kwargs_540176)
        
        # Assigning a type to the variable 'dataset' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'dataset', asarray_call_result_540177)
        
        # Call to check(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_540190 = {}
        
        # Call to FuncData(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'smirnov' (line 89)
        smirnov_540179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'smirnov', False)
        # Getting the type of 'dataset' (line 89)
        dataset_540180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 89)
        tuple_540181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 89)
        # Adding element type (line 89)
        int_540182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 36), tuple_540181, int_540182)
        # Adding element type (line 89)
        int_540183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 36), tuple_540181, int_540183)
        
        int_540184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 43), 'int')
        # Processing the call keyword arguments (line 89)
        float_540185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 51), 'float')
        keyword_540186 = float_540185
        kwargs_540187 = {'rtol': keyword_540186}
        # Getting the type of 'FuncData' (line 89)
        FuncData_540178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 89)
        FuncData_call_result_540188 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), FuncData_540178, *[smirnov_540179, dataset_540180, tuple_540181, int_540184], **kwargs_540187)
        
        # Obtaining the member 'check' of a type (line 89)
        check_540189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), FuncData_call_result_540188, 'check')
        # Calling check(args, kwargs) (line 89)
        check_call_result_540191 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), check_540189, *[], **kwargs_540190)
        
        
        # ################# End of 'test_n_large(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_n_large' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_540192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540192)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_n_large'
        return stypy_return_type_540192


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnov.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSmirnov' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TestSmirnov', TestSmirnov)
# Declaration of the 'TestSmirnovi' class

class TestSmirnovi(object, ):

    @norecursion
    def test_nan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nan'
        module_type_store = module_type_store.open_function_context('test_nan', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_nan')
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_nan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_nan', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nan(...)' code ##################

        
        # Call to assert_(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to isnan(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to smirnovi(...): (line 94)
        # Processing the call arguments (line 94)
        int_540197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 34), 'int')
        # Getting the type of 'np' (line 94)
        np_540198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 37), 'np', False)
        # Obtaining the member 'nan' of a type (line 94)
        nan_540199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 37), np_540198, 'nan')
        # Processing the call keyword arguments (line 94)
        kwargs_540200 = {}
        # Getting the type of 'smirnovi' (line 94)
        smirnovi_540196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'smirnovi', False)
        # Calling smirnovi(args, kwargs) (line 94)
        smirnovi_call_result_540201 = invoke(stypy.reporting.localization.Localization(__file__, 94, 25), smirnovi_540196, *[int_540197, nan_540199], **kwargs_540200)
        
        # Processing the call keyword arguments (line 94)
        kwargs_540202 = {}
        # Getting the type of 'np' (line 94)
        np_540194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'np', False)
        # Obtaining the member 'isnan' of a type (line 94)
        isnan_540195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), np_540194, 'isnan')
        # Calling isnan(args, kwargs) (line 94)
        isnan_call_result_540203 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), isnan_540195, *[smirnovi_call_result_540201], **kwargs_540202)
        
        # Processing the call keyword arguments (line 94)
        kwargs_540204 = {}
        # Getting the type of 'assert_' (line 94)
        assert__540193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 94)
        assert__call_result_540205 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assert__540193, *[isnan_call_result_540203], **kwargs_540204)
        
        
        # ################# End of 'test_nan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nan' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_540206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540206)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nan'
        return stypy_return_type_540206


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_basic')
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a List to a Name (line 98):
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_540207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'tuple' (line 98)
        tuple_540208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 98)
        # Adding element type (line 98)
        int_540209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 20), tuple_540208, int_540209)
        # Adding element type (line 98)
        float_540210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 20), tuple_540208, float_540210)
        # Adding element type (line 98)
        float_540211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 20), tuple_540208, float_540211)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), list_540207, tuple_540208)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_540212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        int_540213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 20), tuple_540212, int_540213)
        # Adding element type (line 99)
        float_540214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 20), tuple_540212, float_540214)
        # Adding element type (line 99)
        float_540215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 20), tuple_540212, float_540215)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), list_540207, tuple_540212)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'tuple' (line 100)
        tuple_540216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 100)
        # Adding element type (line 100)
        int_540217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 20), tuple_540216, int_540217)
        # Adding element type (line 100)
        float_540218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 20), tuple_540216, float_540218)
        # Adding element type (line 100)
        float_540219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 20), tuple_540216, float_540219)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), list_540207, tuple_540216)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_540220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        int_540221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), tuple_540220, int_540221)
        # Adding element type (line 101)
        float_540222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), tuple_540220, float_540222)
        # Adding element type (line 101)
        float_540223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), tuple_540220, float_540223)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), list_540207, tuple_540220)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'tuple' (line 102)
        tuple_540224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 102)
        # Adding element type (line 102)
        int_540225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 20), tuple_540224, int_540225)
        # Adding element type (line 102)
        float_540226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'float')
        float_540227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 31), 'float')
        # Applying the binary operator '*' (line 102)
        result_mul_540228 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 23), '*', float_540226, float_540227)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 20), tuple_540224, result_mul_540228)
        # Adding element type (line 102)
        float_540229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 20), tuple_540224, float_540229)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), list_540207, tuple_540224)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_540230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        int_540231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), tuple_540230, int_540231)
        # Adding element type (line 103)
        float_540232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 23), 'float')
        float_540233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 31), 'float')
        # Applying the binary operator '*' (line 103)
        result_mul_540234 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 23), '*', float_540232, float_540233)
        
        float_540235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 39), 'float')
        # Applying the binary operator '*' (line 103)
        result_mul_540236 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 37), '*', result_mul_540234, float_540235)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), tuple_540230, result_mul_540236)
        # Adding element type (line 103)
        float_540237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), tuple_540230, float_540237)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), list_540207, tuple_540230)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_540238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        int_540239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 20), tuple_540238, int_540239)
        # Adding element type (line 104)
        float_540240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'float')
        int_540241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 30), 'int')
        int_540242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
        # Applying the binary operator '**' (line 104)
        result_pow_540243 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 30), '**', int_540241, int_540242)
        
        # Applying the binary operator 'div' (line 104)
        result_div_540244 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 24), 'div', float_540240, result_pow_540243)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 20), tuple_540238, result_div_540244)
        # Adding element type (line 104)
        int_540245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'int')
        float_540246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 44), 'float')
        int_540247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 50), 'int')
        # Applying the binary operator 'div' (line 104)
        result_div_540248 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 44), 'div', float_540246, int_540247)
        
        # Applying the binary operator '-' (line 104)
        result_sub_540249 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 40), '-', int_540245, result_div_540248)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 20), tuple_540238, result_sub_540249)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), list_540207, tuple_540238)
        
        # Assigning a type to the variable 'dataset' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'dataset', list_540207)
        
        # Assigning a Call to a Name (line 106):
        
        # Call to asarray(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'dataset' (line 106)
        dataset_540252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'dataset', False)
        # Processing the call keyword arguments (line 106)
        kwargs_540253 = {}
        # Getting the type of 'np' (line 106)
        np_540250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 106)
        asarray_540251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 18), np_540250, 'asarray')
        # Calling asarray(args, kwargs) (line 106)
        asarray_call_result_540254 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), asarray_540251, *[dataset_540252], **kwargs_540253)
        
        # Assigning a type to the variable 'dataset' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'dataset', asarray_call_result_540254)
        
        # Call to check(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_540267 = {}
        
        # Call to FuncData(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'smirnovi' (line 107)
        smirnovi_540256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'smirnovi', False)
        # Getting the type of 'dataset' (line 107)
        dataset_540257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 107)
        tuple_540258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 107)
        # Adding element type (line 107)
        int_540259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 37), tuple_540258, int_540259)
        # Adding element type (line 107)
        int_540260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 37), tuple_540258, int_540260)
        
        int_540261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 44), 'int')
        # Processing the call keyword arguments (line 107)
        # Getting the type of '_rtol' (line 107)
        _rtol_540262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 52), '_rtol', False)
        keyword_540263 = _rtol_540262
        kwargs_540264 = {'rtol': keyword_540263}
        # Getting the type of 'FuncData' (line 107)
        FuncData_540255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 107)
        FuncData_call_result_540265 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), FuncData_540255, *[smirnovi_540256, dataset_540257, tuple_540258, int_540261], **kwargs_540264)
        
        # Obtaining the member 'check' of a type (line 107)
        check_540266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), FuncData_call_result_540265, 'check')
        # Calling check(args, kwargs) (line 107)
        check_call_result_540268 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), check_540266, *[], **kwargs_540267)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_540269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_540269


    @norecursion
    def test_x_equals_0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_equals_0'
        module_type_store = module_type_store.open_function_context('test_x_equals_0', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_x_equals_0')
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_x_equals_0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_x_equals_0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_equals_0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_equals_0(...)' code ##################

        
        # Assigning a ListComp to a Name (line 111):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to chain(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to range(...): (line 111)
        # Processing the call arguments (line 111)
        int_540277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 60), 'int')
        int_540278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 63), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_540279 = {}
        # Getting the type of 'range' (line 111)
        range_540276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 54), 'range', False)
        # Calling range(args, kwargs) (line 111)
        range_call_result_540280 = invoke(stypy.reporting.localization.Localization(__file__, 111, 54), range_540276, *[int_540277, int_540278], **kwargs_540279)
        
        
        # Call to range(...): (line 111)
        # Processing the call arguments (line 111)
        int_540282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 74), 'int')
        int_540283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 80), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_540284 = {}
        # Getting the type of 'range' (line 111)
        range_540281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 68), 'range', False)
        # Calling range(args, kwargs) (line 111)
        range_call_result_540285 = invoke(stypy.reporting.localization.Localization(__file__, 111, 68), range_540281, *[int_540282, int_540283], **kwargs_540284)
        
        # Processing the call keyword arguments (line 111)
        kwargs_540286 = {}
        # Getting the type of 'itertools' (line 111)
        itertools_540274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'itertools', False)
        # Obtaining the member 'chain' of a type (line 111)
        chain_540275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 38), itertools_540274, 'chain')
        # Calling chain(args, kwargs) (line 111)
        chain_call_result_540287 = invoke(stypy.reporting.localization.Localization(__file__, 111, 38), chain_540275, *[range_call_result_540280, range_call_result_540285], **kwargs_540286)
        
        comprehension_540288 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), chain_call_result_540287)
        # Assigning a type to the variable 'n' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'n', comprehension_540288)
        
        # Obtaining an instance of the builtin type 'tuple' (line 111)
        tuple_540270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 111)
        # Adding element type (line 111)
        # Getting the type of 'n' (line 111)
        n_540271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 20), tuple_540270, n_540271)
        # Adding element type (line 111)
        int_540272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 20), tuple_540270, int_540272)
        # Adding element type (line 111)
        int_540273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 20), tuple_540270, int_540273)
        
        list_540289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), list_540289, tuple_540270)
        # Assigning a type to the variable 'dataset' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'dataset', list_540289)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to asarray(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'dataset' (line 112)
        dataset_540292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'dataset', False)
        # Processing the call keyword arguments (line 112)
        kwargs_540293 = {}
        # Getting the type of 'np' (line 112)
        np_540290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 112)
        asarray_540291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 18), np_540290, 'asarray')
        # Calling asarray(args, kwargs) (line 112)
        asarray_call_result_540294 = invoke(stypy.reporting.localization.Localization(__file__, 112, 18), asarray_540291, *[dataset_540292], **kwargs_540293)
        
        # Assigning a type to the variable 'dataset' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'dataset', asarray_call_result_540294)
        
        # Call to check(...): (line 113)
        # Processing the call keyword arguments (line 113)
        kwargs_540307 = {}
        
        # Call to FuncData(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'smirnovi' (line 113)
        smirnovi_540296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'smirnovi', False)
        # Getting the type of 'dataset' (line 113)
        dataset_540297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_540298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        int_540299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), tuple_540298, int_540299)
        # Adding element type (line 113)
        int_540300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), tuple_540298, int_540300)
        
        int_540301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'int')
        # Processing the call keyword arguments (line 113)
        # Getting the type of '_rtol' (line 113)
        _rtol_540302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 52), '_rtol', False)
        keyword_540303 = _rtol_540302
        kwargs_540304 = {'rtol': keyword_540303}
        # Getting the type of 'FuncData' (line 113)
        FuncData_540295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 113)
        FuncData_call_result_540305 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), FuncData_540295, *[smirnovi_540296, dataset_540297, tuple_540298, int_540301], **kwargs_540304)
        
        # Obtaining the member 'check' of a type (line 113)
        check_540306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), FuncData_call_result_540305, 'check')
        # Calling check(args, kwargs) (line 113)
        check_call_result_540308 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), check_540306, *[], **kwargs_540307)
        
        
        # ################# End of 'test_x_equals_0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_equals_0' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_540309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540309)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_equals_0'
        return stypy_return_type_540309


    @norecursion
    def test_x_equals_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_equals_1'
        module_type_store = module_type_store.open_function_context('test_x_equals_1', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_x_equals_1')
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_x_equals_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_x_equals_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_equals_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_equals_1(...)' code ##################

        
        # Assigning a ListComp to a Name (line 116):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to chain(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to range(...): (line 116)
        # Processing the call arguments (line 116)
        int_540317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 60), 'int')
        int_540318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 63), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_540319 = {}
        # Getting the type of 'range' (line 116)
        range_540316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 54), 'range', False)
        # Calling range(args, kwargs) (line 116)
        range_call_result_540320 = invoke(stypy.reporting.localization.Localization(__file__, 116, 54), range_540316, *[int_540317, int_540318], **kwargs_540319)
        
        
        # Call to range(...): (line 116)
        # Processing the call arguments (line 116)
        int_540322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 74), 'int')
        int_540323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 80), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_540324 = {}
        # Getting the type of 'range' (line 116)
        range_540321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 68), 'range', False)
        # Calling range(args, kwargs) (line 116)
        range_call_result_540325 = invoke(stypy.reporting.localization.Localization(__file__, 116, 68), range_540321, *[int_540322, int_540323], **kwargs_540324)
        
        # Processing the call keyword arguments (line 116)
        kwargs_540326 = {}
        # Getting the type of 'itertools' (line 116)
        itertools_540314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 38), 'itertools', False)
        # Obtaining the member 'chain' of a type (line 116)
        chain_540315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 38), itertools_540314, 'chain')
        # Calling chain(args, kwargs) (line 116)
        chain_call_result_540327 = invoke(stypy.reporting.localization.Localization(__file__, 116, 38), chain_540315, *[range_call_result_540320, range_call_result_540325], **kwargs_540326)
        
        comprehension_540328 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), chain_call_result_540327)
        # Assigning a type to the variable 'n' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'n', comprehension_540328)
        
        # Obtaining an instance of the builtin type 'tuple' (line 116)
        tuple_540310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 116)
        # Adding element type (line 116)
        # Getting the type of 'n' (line 116)
        n_540311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), tuple_540310, n_540311)
        # Adding element type (line 116)
        int_540312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), tuple_540310, int_540312)
        # Adding element type (line 116)
        int_540313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), tuple_540310, int_540313)
        
        list_540329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_540329, tuple_540310)
        # Assigning a type to the variable 'dataset' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'dataset', list_540329)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to asarray(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'dataset' (line 117)
        dataset_540332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'dataset', False)
        # Processing the call keyword arguments (line 117)
        kwargs_540333 = {}
        # Getting the type of 'np' (line 117)
        np_540330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 117)
        asarray_540331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 18), np_540330, 'asarray')
        # Calling asarray(args, kwargs) (line 117)
        asarray_call_result_540334 = invoke(stypy.reporting.localization.Localization(__file__, 117, 18), asarray_540331, *[dataset_540332], **kwargs_540333)
        
        # Assigning a type to the variable 'dataset' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'dataset', asarray_call_result_540334)
        
        # Call to check(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_540347 = {}
        
        # Call to FuncData(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'smirnovi' (line 118)
        smirnovi_540336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'smirnovi', False)
        # Getting the type of 'dataset' (line 118)
        dataset_540337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_540338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        int_540339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 37), tuple_540338, int_540339)
        # Adding element type (line 118)
        int_540340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 37), tuple_540338, int_540340)
        
        int_540341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 44), 'int')
        # Processing the call keyword arguments (line 118)
        # Getting the type of '_rtol' (line 118)
        _rtol_540342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 52), '_rtol', False)
        keyword_540343 = _rtol_540342
        kwargs_540344 = {'rtol': keyword_540343}
        # Getting the type of 'FuncData' (line 118)
        FuncData_540335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 118)
        FuncData_call_result_540345 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), FuncData_540335, *[smirnovi_540336, dataset_540337, tuple_540338, int_540341], **kwargs_540344)
        
        # Obtaining the member 'check' of a type (line 118)
        check_540346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), FuncData_call_result_540345, 'check')
        # Calling check(args, kwargs) (line 118)
        check_call_result_540348 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), check_540346, *[], **kwargs_540347)
        
        
        # ################# End of 'test_x_equals_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_equals_1' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_540349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540349)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_equals_1'
        return stypy_return_type_540349


    @norecursion
    def test_n_equals_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_n_equals_1'
        module_type_store = module_type_store.open_function_context('test_n_equals_1', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_n_equals_1')
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_n_equals_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_n_equals_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_n_equals_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_n_equals_1(...)' code ##################

        
        # Assigning a Call to a Name (line 122):
        
        # Call to linspace(...): (line 122)
        # Processing the call arguments (line 122)
        int_540352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 25), 'int')
        int_540353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 28), 'int')
        int_540354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 31), 'int')
        # Processing the call keyword arguments (line 122)
        # Getting the type of 'True' (line 122)
        True_540355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'True', False)
        keyword_540356 = True_540355
        kwargs_540357 = {'endpoint': keyword_540356}
        # Getting the type of 'np' (line 122)
        np_540350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'np', False)
        # Obtaining the member 'linspace' of a type (line 122)
        linspace_540351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 13), np_540350, 'linspace')
        # Calling linspace(args, kwargs) (line 122)
        linspace_call_result_540358 = invoke(stypy.reporting.localization.Localization(__file__, 122, 13), linspace_540351, *[int_540352, int_540353, int_540354], **kwargs_540357)
        
        # Assigning a type to the variable 'pp' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'pp', linspace_call_result_540358)
        
        # Assigning a ListComp to a Name (line 123):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'pp' (line 123)
        pp_540365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 40), 'pp')
        comprehension_540366 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 19), pp_540365)
        # Assigning a type to the variable 'p' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'p', comprehension_540366)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_540359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        int_540360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), tuple_540359, int_540360)
        # Adding element type (line 123)
        # Getting the type of 'p' (line 123)
        p_540361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), tuple_540359, p_540361)
        # Adding element type (line 123)
        int_540362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 26), 'int')
        # Getting the type of 'p' (line 123)
        p_540363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'p')
        # Applying the binary operator '-' (line 123)
        result_sub_540364 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 26), '-', int_540362, p_540363)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), tuple_540359, result_sub_540364)
        
        list_540367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 19), list_540367, tuple_540359)
        # Assigning a type to the variable 'dataset' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'dataset', list_540367)
        
        # Assigning a Call to a Name (line 124):
        
        # Call to asarray(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'dataset' (line 124)
        dataset_540370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'dataset', False)
        # Processing the call keyword arguments (line 124)
        kwargs_540371 = {}
        # Getting the type of 'np' (line 124)
        np_540368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 124)
        asarray_540369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 18), np_540368, 'asarray')
        # Calling asarray(args, kwargs) (line 124)
        asarray_call_result_540372 = invoke(stypy.reporting.localization.Localization(__file__, 124, 18), asarray_540369, *[dataset_540370], **kwargs_540371)
        
        # Assigning a type to the variable 'dataset' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'dataset', asarray_call_result_540372)
        
        # Call to check(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_540385 = {}
        
        # Call to FuncData(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'smirnovi' (line 125)
        smirnovi_540374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'smirnovi', False)
        # Getting the type of 'dataset' (line 125)
        dataset_540375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_540376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        int_540377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 37), tuple_540376, int_540377)
        # Adding element type (line 125)
        int_540378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 37), tuple_540376, int_540378)
        
        int_540379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 44), 'int')
        # Processing the call keyword arguments (line 125)
        # Getting the type of '_rtol' (line 125)
        _rtol_540380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 52), '_rtol', False)
        keyword_540381 = _rtol_540380
        kwargs_540382 = {'rtol': keyword_540381}
        # Getting the type of 'FuncData' (line 125)
        FuncData_540373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 125)
        FuncData_call_result_540383 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), FuncData_540373, *[smirnovi_540374, dataset_540375, tuple_540376, int_540379], **kwargs_540382)
        
        # Obtaining the member 'check' of a type (line 125)
        check_540384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), FuncData_call_result_540383, 'check')
        # Calling check(args, kwargs) (line 125)
        check_call_result_540386 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), check_540384, *[], **kwargs_540385)
        
        
        # ################# End of 'test_n_equals_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_n_equals_1' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_540387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_n_equals_1'
        return stypy_return_type_540387


    @norecursion
    def test_n_equals_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_n_equals_2'
        module_type_store = module_type_store.open_function_context('test_n_equals_2', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_n_equals_2')
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_n_equals_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_n_equals_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_n_equals_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_n_equals_2(...)' code ##################

        
        # Assigning a Call to a Name (line 129):
        
        # Call to linspace(...): (line 129)
        # Processing the call arguments (line 129)
        float_540390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'float')
        int_540391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 29), 'int')
        int_540392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 32), 'int')
        # Processing the call keyword arguments (line 129)
        # Getting the type of 'True' (line 129)
        True_540393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 46), 'True', False)
        keyword_540394 = True_540393
        kwargs_540395 = {'endpoint': keyword_540394}
        # Getting the type of 'np' (line 129)
        np_540388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 129)
        linspace_540389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), np_540388, 'linspace')
        # Calling linspace(args, kwargs) (line 129)
        linspace_call_result_540396 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), linspace_540389, *[float_540390, int_540391, int_540392], **kwargs_540395)
        
        # Assigning a type to the variable 'x' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'x', linspace_call_result_540396)
        
        # Assigning a Call to a Name (line 130):
        
        # Call to power(...): (line 130)
        # Processing the call arguments (line 130)
        int_540399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'int')
        # Getting the type of 'x' (line 130)
        x_540400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 23), 'x', False)
        # Applying the binary operator '-' (line 130)
        result_sub_540401 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 21), '-', int_540399, x_540400)
        
        int_540402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 26), 'int')
        # Processing the call keyword arguments (line 130)
        kwargs_540403 = {}
        # Getting the type of 'np' (line 130)
        np_540397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'np', False)
        # Obtaining the member 'power' of a type (line 130)
        power_540398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), np_540397, 'power')
        # Calling power(args, kwargs) (line 130)
        power_call_result_540404 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), power_540398, *[result_sub_540401, int_540402], **kwargs_540403)
        
        # Assigning a type to the variable 'p' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'p', power_call_result_540404)
        
        # Assigning a Call to a Name (line 131):
        
        # Call to array(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Obtaining an instance of the builtin type 'list' (line 131)
        list_540407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 131)
        # Adding element type (line 131)
        int_540408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 21), list_540407, int_540408)
        
        
        # Call to len(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'x' (line 131)
        x_540410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'x', False)
        # Processing the call keyword arguments (line 131)
        kwargs_540411 = {}
        # Getting the type of 'len' (line 131)
        len_540409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'len', False)
        # Calling len(args, kwargs) (line 131)
        len_call_result_540412 = invoke(stypy.reporting.localization.Localization(__file__, 131, 27), len_540409, *[x_540410], **kwargs_540411)
        
        # Applying the binary operator '*' (line 131)
        result_mul_540413 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 21), '*', list_540407, len_call_result_540412)
        
        # Processing the call keyword arguments (line 131)
        kwargs_540414 = {}
        # Getting the type of 'np' (line 131)
        np_540405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 131)
        array_540406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), np_540405, 'array')
        # Calling array(args, kwargs) (line 131)
        array_call_result_540415 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), array_540406, *[result_mul_540413], **kwargs_540414)
        
        # Assigning a type to the variable 'n' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'n', array_call_result_540415)
        
        # Assigning a Call to a Name (line 132):
        
        # Call to column_stack(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_540418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        # Getting the type of 'n' (line 132)
        n_540419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 34), list_540418, n_540419)
        # Adding element type (line 132)
        # Getting the type of 'p' (line 132)
        p_540420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 38), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 34), list_540418, p_540420)
        # Adding element type (line 132)
        # Getting the type of 'x' (line 132)
        x_540421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 41), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 34), list_540418, x_540421)
        
        # Processing the call keyword arguments (line 132)
        kwargs_540422 = {}
        # Getting the type of 'np' (line 132)
        np_540416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 132)
        column_stack_540417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 18), np_540416, 'column_stack')
        # Calling column_stack(args, kwargs) (line 132)
        column_stack_call_result_540423 = invoke(stypy.reporting.localization.Localization(__file__, 132, 18), column_stack_540417, *[list_540418], **kwargs_540422)
        
        # Assigning a type to the variable 'dataset' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'dataset', column_stack_call_result_540423)
        
        # Call to check(...): (line 134)
        # Processing the call keyword arguments (line 134)
        kwargs_540436 = {}
        
        # Call to FuncData(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'smirnovi' (line 134)
        smirnovi_540425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'smirnovi', False)
        # Getting the type of 'dataset' (line 134)
        dataset_540426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 134)
        tuple_540427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 134)
        # Adding element type (line 134)
        int_540428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 37), tuple_540427, int_540428)
        # Adding element type (line 134)
        int_540429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 37), tuple_540427, int_540429)
        
        int_540430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 44), 'int')
        # Processing the call keyword arguments (line 134)
        # Getting the type of '_rtol' (line 134)
        _rtol_540431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 52), '_rtol', False)
        keyword_540432 = _rtol_540431
        kwargs_540433 = {'rtol': keyword_540432}
        # Getting the type of 'FuncData' (line 134)
        FuncData_540424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 134)
        FuncData_call_result_540434 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), FuncData_540424, *[smirnovi_540425, dataset_540426, tuple_540427, int_540430], **kwargs_540433)
        
        # Obtaining the member 'check' of a type (line 134)
        check_540435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), FuncData_call_result_540434, 'check')
        # Calling check(args, kwargs) (line 134)
        check_call_result_540437 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), check_540435, *[], **kwargs_540436)
        
        
        # ################# End of 'test_n_equals_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_n_equals_2' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_540438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540438)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_n_equals_2'
        return stypy_return_type_540438


    @norecursion
    def test_n_equals_3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_n_equals_3'
        module_type_store = module_type_store.open_function_context('test_n_equals_3', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_n_equals_3')
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_n_equals_3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_n_equals_3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_n_equals_3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_n_equals_3(...)' code ##################

        
        # Assigning a Call to a Name (line 138):
        
        # Call to linspace(...): (line 138)
        # Processing the call arguments (line 138)
        float_540441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'float')
        int_540442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'int')
        int_540443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 32), 'int')
        # Processing the call keyword arguments (line 138)
        # Getting the type of 'True' (line 138)
        True_540444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 45), 'True', False)
        keyword_540445 = True_540444
        kwargs_540446 = {'endpoint': keyword_540445}
        # Getting the type of 'np' (line 138)
        np_540439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 138)
        linspace_540440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), np_540439, 'linspace')
        # Calling linspace(args, kwargs) (line 138)
        linspace_call_result_540447 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), linspace_540440, *[float_540441, int_540442, int_540443], **kwargs_540446)
        
        # Assigning a type to the variable 'x' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'x', linspace_call_result_540447)
        
        # Assigning a Call to a Name (line 139):
        
        # Call to power(...): (line 139)
        # Processing the call arguments (line 139)
        int_540450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'int')
        # Getting the type of 'x' (line 139)
        x_540451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'x', False)
        # Applying the binary operator '-' (line 139)
        result_sub_540452 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), '-', int_540450, x_540451)
        
        int_540453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 26), 'int')
        # Processing the call keyword arguments (line 139)
        kwargs_540454 = {}
        # Getting the type of 'np' (line 139)
        np_540448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'np', False)
        # Obtaining the member 'power' of a type (line 139)
        power_540449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), np_540448, 'power')
        # Calling power(args, kwargs) (line 139)
        power_call_result_540455 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), power_540449, *[result_sub_540452, int_540453], **kwargs_540454)
        
        # Assigning a type to the variable 'p' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'p', power_call_result_540455)
        
        # Assigning a Call to a Name (line 140):
        
        # Call to array(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_540458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        int_540459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), list_540458, int_540459)
        
        
        # Call to len(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'x' (line 140)
        x_540461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 31), 'x', False)
        # Processing the call keyword arguments (line 140)
        kwargs_540462 = {}
        # Getting the type of 'len' (line 140)
        len_540460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'len', False)
        # Calling len(args, kwargs) (line 140)
        len_call_result_540463 = invoke(stypy.reporting.localization.Localization(__file__, 140, 27), len_540460, *[x_540461], **kwargs_540462)
        
        # Applying the binary operator '*' (line 140)
        result_mul_540464 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 21), '*', list_540458, len_call_result_540463)
        
        # Processing the call keyword arguments (line 140)
        kwargs_540465 = {}
        # Getting the type of 'np' (line 140)
        np_540456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 140)
        array_540457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), np_540456, 'array')
        # Calling array(args, kwargs) (line 140)
        array_call_result_540466 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), array_540457, *[result_mul_540464], **kwargs_540465)
        
        # Assigning a type to the variable 'n' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'n', array_call_result_540466)
        
        # Assigning a Call to a Name (line 141):
        
        # Call to column_stack(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Obtaining an instance of the builtin type 'list' (line 141)
        list_540469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 141)
        # Adding element type (line 141)
        # Getting the type of 'n' (line 141)
        n_540470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 34), list_540469, n_540470)
        # Adding element type (line 141)
        # Getting the type of 'p' (line 141)
        p_540471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 38), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 34), list_540469, p_540471)
        # Adding element type (line 141)
        # Getting the type of 'x' (line 141)
        x_540472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 34), list_540469, x_540472)
        
        # Processing the call keyword arguments (line 141)
        kwargs_540473 = {}
        # Getting the type of 'np' (line 141)
        np_540467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 141)
        column_stack_540468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 18), np_540467, 'column_stack')
        # Calling column_stack(args, kwargs) (line 141)
        column_stack_call_result_540474 = invoke(stypy.reporting.localization.Localization(__file__, 141, 18), column_stack_540468, *[list_540469], **kwargs_540473)
        
        # Assigning a type to the variable 'dataset' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'dataset', column_stack_call_result_540474)
        
        # Call to check(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_540487 = {}
        
        # Call to FuncData(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'smirnovi' (line 143)
        smirnovi_540476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'smirnovi', False)
        # Getting the type of 'dataset' (line 143)
        dataset_540477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 143)
        tuple_540478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 143)
        # Adding element type (line 143)
        int_540479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 37), tuple_540478, int_540479)
        # Adding element type (line 143)
        int_540480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 37), tuple_540478, int_540480)
        
        int_540481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 44), 'int')
        # Processing the call keyword arguments (line 143)
        # Getting the type of '_rtol' (line 143)
        _rtol_540482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 52), '_rtol', False)
        keyword_540483 = _rtol_540482
        kwargs_540484 = {'rtol': keyword_540483}
        # Getting the type of 'FuncData' (line 143)
        FuncData_540475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 143)
        FuncData_call_result_540485 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), FuncData_540475, *[smirnovi_540476, dataset_540477, tuple_540478, int_540481], **kwargs_540484)
        
        # Obtaining the member 'check' of a type (line 143)
        check_540486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), FuncData_call_result_540485, 'check')
        # Calling check(args, kwargs) (line 143)
        check_call_result_540488 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), check_540486, *[], **kwargs_540487)
        
        
        # ################# End of 'test_n_equals_3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_n_equals_3' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_540489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_n_equals_3'
        return stypy_return_type_540489


    @norecursion
    def test_round_trip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_round_trip'
        module_type_store = module_type_store.open_function_context('test_round_trip', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_round_trip')
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_round_trip.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_round_trip', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_round_trip', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_round_trip(...)' code ##################


        @norecursion
        def _sm_smi(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_sm_smi'
            module_type_store = module_type_store.open_function_context('_sm_smi', 147, 8, False)
            
            # Passed parameters checking function
            _sm_smi.stypy_localization = localization
            _sm_smi.stypy_type_of_self = None
            _sm_smi.stypy_type_store = module_type_store
            _sm_smi.stypy_function_name = '_sm_smi'
            _sm_smi.stypy_param_names_list = ['n', 'p']
            _sm_smi.stypy_varargs_param_name = None
            _sm_smi.stypy_kwargs_param_name = None
            _sm_smi.stypy_call_defaults = defaults
            _sm_smi.stypy_call_varargs = varargs
            _sm_smi.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_sm_smi', ['n', 'p'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_sm_smi', localization, ['n', 'p'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_sm_smi(...)' code ##################

            
            # Call to smirnov(...): (line 148)
            # Processing the call arguments (line 148)
            # Getting the type of 'n' (line 148)
            n_540491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'n', False)
            
            # Call to smirnovi(...): (line 148)
            # Processing the call arguments (line 148)
            # Getting the type of 'n' (line 148)
            n_540493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'n', False)
            # Getting the type of 'p' (line 148)
            p_540494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 42), 'p', False)
            # Processing the call keyword arguments (line 148)
            kwargs_540495 = {}
            # Getting the type of 'smirnovi' (line 148)
            smirnovi_540492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'smirnovi', False)
            # Calling smirnovi(args, kwargs) (line 148)
            smirnovi_call_result_540496 = invoke(stypy.reporting.localization.Localization(__file__, 148, 30), smirnovi_540492, *[n_540493, p_540494], **kwargs_540495)
            
            # Processing the call keyword arguments (line 148)
            kwargs_540497 = {}
            # Getting the type of 'smirnov' (line 148)
            smirnov_540490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'smirnov', False)
            # Calling smirnov(args, kwargs) (line 148)
            smirnov_call_result_540498 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), smirnov_540490, *[n_540491, smirnovi_call_result_540496], **kwargs_540497)
            
            # Assigning a type to the variable 'stypy_return_type' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'stypy_return_type', smirnov_call_result_540498)
            
            # ################# End of '_sm_smi(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_sm_smi' in the type store
            # Getting the type of 'stypy_return_type' (line 147)
            stypy_return_type_540499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_540499)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_sm_smi'
            return stypy_return_type_540499

        # Assigning a type to the variable '_sm_smi' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), '_sm_smi', _sm_smi)
        
        # Assigning a List to a Name (line 150):
        
        # Obtaining an instance of the builtin type 'list' (line 150)
        list_540500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 150)
        # Adding element type (line 150)
        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_540501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        int_540502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 20), tuple_540501, int_540502)
        # Adding element type (line 150)
        float_540503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 20), tuple_540501, float_540503)
        # Adding element type (line 150)
        float_540504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 20), tuple_540501, float_540504)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_540500, tuple_540501)
        # Adding element type (line 150)
        
        # Obtaining an instance of the builtin type 'tuple' (line 151)
        tuple_540505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 151)
        # Adding element type (line 151)
        int_540506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 20), tuple_540505, int_540506)
        # Adding element type (line 151)
        float_540507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 20), tuple_540505, float_540507)
        # Adding element type (line 151)
        float_540508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 20), tuple_540505, float_540508)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_540500, tuple_540505)
        # Adding element type (line 150)
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_540509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        int_540510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 20), tuple_540509, int_540510)
        # Adding element type (line 152)
        float_540511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 20), tuple_540509, float_540511)
        # Adding element type (line 152)
        float_540512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 20), tuple_540509, float_540512)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_540500, tuple_540509)
        # Adding element type (line 150)
        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_540513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        int_540514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 20), tuple_540513, int_540514)
        # Adding element type (line 153)
        float_540515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 20), tuple_540513, float_540515)
        # Adding element type (line 153)
        float_540516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 20), tuple_540513, float_540516)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_540500, tuple_540513)
        # Adding element type (line 150)
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_540517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        int_540518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), tuple_540517, int_540518)
        # Adding element type (line 154)
        float_540519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), tuple_540517, float_540519)
        # Adding element type (line 154)
        float_540520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), tuple_540517, float_540520)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_540500, tuple_540517)
        # Adding element type (line 150)
        
        # Obtaining an instance of the builtin type 'tuple' (line 155)
        tuple_540521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 155)
        # Adding element type (line 155)
        int_540522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 20), tuple_540521, int_540522)
        # Adding element type (line 155)
        float_540523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 20), tuple_540521, float_540523)
        # Adding element type (line 155)
        float_540524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 20), tuple_540521, float_540524)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_540500, tuple_540521)
        # Adding element type (line 150)
        
        # Obtaining an instance of the builtin type 'tuple' (line 156)
        tuple_540525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 156)
        # Adding element type (line 156)
        int_540526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), tuple_540525, int_540526)
        # Adding element type (line 156)
        float_540527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), tuple_540525, float_540527)
        # Adding element type (line 156)
        float_540528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), tuple_540525, float_540528)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_540500, tuple_540525)
        
        # Assigning a type to the variable 'dataset' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'dataset', list_540500)
        
        # Assigning a Call to a Name (line 158):
        
        # Call to asarray(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'dataset' (line 158)
        dataset_540531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'dataset', False)
        # Processing the call keyword arguments (line 158)
        kwargs_540532 = {}
        # Getting the type of 'np' (line 158)
        np_540529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 158)
        asarray_540530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 18), np_540529, 'asarray')
        # Calling asarray(args, kwargs) (line 158)
        asarray_call_result_540533 = invoke(stypy.reporting.localization.Localization(__file__, 158, 18), asarray_540530, *[dataset_540531], **kwargs_540532)
        
        # Assigning a type to the variable 'dataset' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'dataset', asarray_call_result_540533)
        
        # Call to check(...): (line 159)
        # Processing the call keyword arguments (line 159)
        kwargs_540546 = {}
        
        # Call to FuncData(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of '_sm_smi' (line 159)
        _sm_smi_540535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), '_sm_smi', False)
        # Getting the type of 'dataset' (line 159)
        dataset_540536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 159)
        tuple_540537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 159)
        # Adding element type (line 159)
        int_540538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 36), tuple_540537, int_540538)
        # Adding element type (line 159)
        int_540539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 36), tuple_540537, int_540539)
        
        int_540540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'int')
        # Processing the call keyword arguments (line 159)
        # Getting the type of '_rtol' (line 159)
        _rtol_540541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), '_rtol', False)
        keyword_540542 = _rtol_540541
        kwargs_540543 = {'rtol': keyword_540542}
        # Getting the type of 'FuncData' (line 159)
        FuncData_540534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 159)
        FuncData_call_result_540544 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), FuncData_540534, *[_sm_smi_540535, dataset_540536, tuple_540537, int_540540], **kwargs_540543)
        
        # Obtaining the member 'check' of a type (line 159)
        check_540545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), FuncData_call_result_540544, 'check')
        # Calling check(args, kwargs) (line 159)
        check_call_result_540547 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), check_540545, *[], **kwargs_540546)
        
        
        # ################# End of 'test_round_trip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_round_trip' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_540548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540548)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_round_trip'
        return stypy_return_type_540548


    @norecursion
    def test_x_equals_0point5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_equals_0point5'
        module_type_store = module_type_store.open_function_context('test_x_equals_0point5', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_localization', localization)
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_function_name', 'TestSmirnovi.test_x_equals_0point5')
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmirnovi.test_x_equals_0point5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.test_x_equals_0point5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_equals_0point5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_equals_0point5(...)' code ##################

        
        # Assigning a List to a Name (line 162):
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_540549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 162)
        tuple_540550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 162)
        # Adding element type (line 162)
        int_540551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 20), tuple_540550, int_540551)
        # Adding element type (line 162)
        float_540552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 20), tuple_540550, float_540552)
        # Adding element type (line 162)
        float_540553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 20), tuple_540550, float_540553)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540550)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 163)
        tuple_540554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 163)
        # Adding element type (line 163)
        int_540555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 20), tuple_540554, int_540555)
        # Adding element type (line 163)
        float_540556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 20), tuple_540554, float_540556)
        # Adding element type (line 163)
        float_540557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 20), tuple_540554, float_540557)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540554)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 164)
        tuple_540558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 164)
        # Adding element type (line 164)
        int_540559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 20), tuple_540558, int_540559)
        # Adding element type (line 164)
        float_540560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 20), tuple_540558, float_540560)
        # Adding element type (line 164)
        float_540561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 20), tuple_540558, float_540561)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540558)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 165)
        tuple_540562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 165)
        # Adding element type (line 165)
        int_540563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_540562, int_540563)
        # Adding element type (line 165)
        float_540564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_540562, float_540564)
        # Adding element type (line 165)
        float_540565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_540562, float_540565)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540562)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 166)
        tuple_540566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 166)
        # Adding element type (line 166)
        int_540567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 20), tuple_540566, int_540567)
        # Adding element type (line 166)
        float_540568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 20), tuple_540566, float_540568)
        # Adding element type (line 166)
        float_540569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 20), tuple_540566, float_540569)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540566)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 167)
        tuple_540570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 167)
        # Adding element type (line 167)
        int_540571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), tuple_540570, int_540571)
        # Adding element type (line 167)
        float_540572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), tuple_540570, float_540572)
        # Adding element type (line 167)
        float_540573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), tuple_540570, float_540573)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540570)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 168)
        tuple_540574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 168)
        # Adding element type (line 168)
        int_540575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), tuple_540574, int_540575)
        # Adding element type (line 168)
        float_540576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), tuple_540574, float_540576)
        # Adding element type (line 168)
        float_540577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), tuple_540574, float_540577)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540574)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_540578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        int_540579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), tuple_540578, int_540579)
        # Adding element type (line 169)
        float_540580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), tuple_540578, float_540580)
        # Adding element type (line 169)
        float_540581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), tuple_540578, float_540581)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540578)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 170)
        tuple_540582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 170)
        # Adding element type (line 170)
        int_540583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 20), tuple_540582, int_540583)
        # Adding element type (line 170)
        float_540584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 20), tuple_540582, float_540584)
        # Adding element type (line 170)
        float_540585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 20), tuple_540582, float_540585)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540582)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 171)
        tuple_540586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 171)
        # Adding element type (line 171)
        int_540587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 20), tuple_540586, int_540587)
        # Adding element type (line 171)
        float_540588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 20), tuple_540586, float_540588)
        # Adding element type (line 171)
        float_540589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 20), tuple_540586, float_540589)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540586)
        # Adding element type (line 162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 172)
        tuple_540590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 172)
        # Adding element type (line 172)
        int_540591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 20), tuple_540590, int_540591)
        # Adding element type (line 172)
        float_540592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 20), tuple_540590, float_540592)
        # Adding element type (line 172)
        float_540593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 20), tuple_540590, float_540593)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 18), list_540549, tuple_540590)
        
        # Assigning a type to the variable 'dataset' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'dataset', list_540549)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to asarray(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'dataset' (line 174)
        dataset_540596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'dataset', False)
        # Processing the call keyword arguments (line 174)
        kwargs_540597 = {}
        # Getting the type of 'np' (line 174)
        np_540594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 174)
        asarray_540595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 18), np_540594, 'asarray')
        # Calling asarray(args, kwargs) (line 174)
        asarray_call_result_540598 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), asarray_540595, *[dataset_540596], **kwargs_540597)
        
        # Assigning a type to the variable 'dataset' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'dataset', asarray_call_result_540598)
        
        # Call to check(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_540611 = {}
        
        # Call to FuncData(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'smirnovi' (line 175)
        smirnovi_540600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'smirnovi', False)
        # Getting the type of 'dataset' (line 175)
        dataset_540601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 27), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 175)
        tuple_540602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 175)
        # Adding element type (line 175)
        int_540603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 37), tuple_540602, int_540603)
        # Adding element type (line 175)
        int_540604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 37), tuple_540602, int_540604)
        
        int_540605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 44), 'int')
        # Processing the call keyword arguments (line 175)
        # Getting the type of '_rtol' (line 175)
        _rtol_540606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 52), '_rtol', False)
        keyword_540607 = _rtol_540606
        kwargs_540608 = {'rtol': keyword_540607}
        # Getting the type of 'FuncData' (line 175)
        FuncData_540599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 175)
        FuncData_call_result_540609 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), FuncData_540599, *[smirnovi_540600, dataset_540601, tuple_540602, int_540605], **kwargs_540608)
        
        # Obtaining the member 'check' of a type (line 175)
        check_540610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), FuncData_call_result_540609, 'check')
        # Calling check(args, kwargs) (line 175)
        check_call_result_540612 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), check_540610, *[], **kwargs_540611)
        
        
        # ################# End of 'test_x_equals_0point5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_equals_0point5' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_540613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540613)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_equals_0point5'
        return stypy_return_type_540613


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 92, 0, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmirnovi.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSmirnovi' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'TestSmirnovi', TestSmirnovi)
# Declaration of the 'TestKolmogorov' class

class TestKolmogorov(object, ):

    @norecursion
    def test_nan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nan'
        module_type_store = module_type_store.open_function_context('test_nan', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_localization', localization)
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_function_name', 'TestKolmogorov.test_nan')
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_param_names_list', [])
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKolmogorov.test_nan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogorov.test_nan', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nan(...)' code ##################

        
        # Call to assert_(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to isnan(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to kolmogorov(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'np' (line 180)
        np_540618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 36), 'np', False)
        # Obtaining the member 'nan' of a type (line 180)
        nan_540619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 36), np_540618, 'nan')
        # Processing the call keyword arguments (line 180)
        kwargs_540620 = {}
        # Getting the type of 'kolmogorov' (line 180)
        kolmogorov_540617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'kolmogorov', False)
        # Calling kolmogorov(args, kwargs) (line 180)
        kolmogorov_call_result_540621 = invoke(stypy.reporting.localization.Localization(__file__, 180, 25), kolmogorov_540617, *[nan_540619], **kwargs_540620)
        
        # Processing the call keyword arguments (line 180)
        kwargs_540622 = {}
        # Getting the type of 'np' (line 180)
        np_540615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'np', False)
        # Obtaining the member 'isnan' of a type (line 180)
        isnan_540616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 16), np_540615, 'isnan')
        # Calling isnan(args, kwargs) (line 180)
        isnan_call_result_540623 = invoke(stypy.reporting.localization.Localization(__file__, 180, 16), isnan_540616, *[kolmogorov_call_result_540621], **kwargs_540622)
        
        # Processing the call keyword arguments (line 180)
        kwargs_540624 = {}
        # Getting the type of 'assert_' (line 180)
        assert__540614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 180)
        assert__call_result_540625 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), assert__540614, *[isnan_call_result_540623], **kwargs_540624)
        
        
        # ################# End of 'test_nan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nan' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_540626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540626)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nan'
        return stypy_return_type_540626


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_function_name', 'TestKolmogorov.test_basic')
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKolmogorov.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogorov.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a List to a Name (line 183):
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_540627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 183)
        tuple_540628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 183)
        # Adding element type (line 183)
        int_540629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 20), tuple_540628, int_540629)
        # Adding element type (line 183)
        float_540630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 20), tuple_540628, float_540630)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 18), list_540627, tuple_540628)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_540631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        float_540632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), tuple_540631, float_540632)
        # Adding element type (line 184)
        float_540633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), tuple_540631, float_540633)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 18), list_540627, tuple_540631)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_540634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_540635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), tuple_540634, int_540635)
        # Adding element type (line 185)
        float_540636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), tuple_540634, float_540636)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 18), list_540627, tuple_540634)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_540637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)
        int_540638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 20), tuple_540637, int_540638)
        # Adding element type (line 186)
        float_540639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 20), tuple_540637, float_540639)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 18), list_540627, tuple_540637)
        
        # Assigning a type to the variable 'dataset' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'dataset', list_540627)
        
        # Assigning a Call to a Name (line 188):
        
        # Call to asarray(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'dataset' (line 188)
        dataset_540642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 29), 'dataset', False)
        # Processing the call keyword arguments (line 188)
        kwargs_540643 = {}
        # Getting the type of 'np' (line 188)
        np_540640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 188)
        asarray_540641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 18), np_540640, 'asarray')
        # Calling asarray(args, kwargs) (line 188)
        asarray_call_result_540644 = invoke(stypy.reporting.localization.Localization(__file__, 188, 18), asarray_540641, *[dataset_540642], **kwargs_540643)
        
        # Assigning a type to the variable 'dataset' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'dataset', asarray_call_result_540644)
        
        # Call to check(...): (line 189)
        # Processing the call keyword arguments (line 189)
        kwargs_540656 = {}
        
        # Call to FuncData(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'kolmogorov' (line 189)
        kolmogorov_540646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'kolmogorov', False)
        # Getting the type of 'dataset' (line 189)
        dataset_540647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 189)
        tuple_540648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 189)
        # Adding element type (line 189)
        int_540649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 39), tuple_540648, int_540649)
        
        int_540650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 44), 'int')
        # Processing the call keyword arguments (line 189)
        # Getting the type of '_rtol' (line 189)
        _rtol_540651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 52), '_rtol', False)
        keyword_540652 = _rtol_540651
        kwargs_540653 = {'rtol': keyword_540652}
        # Getting the type of 'FuncData' (line 189)
        FuncData_540645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 189)
        FuncData_call_result_540654 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), FuncData_540645, *[kolmogorov_540646, dataset_540647, tuple_540648, int_540650], **kwargs_540653)
        
        # Obtaining the member 'check' of a type (line 189)
        check_540655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), FuncData_call_result_540654, 'check')
        # Calling check(args, kwargs) (line 189)
        check_call_result_540657 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), check_540655, *[], **kwargs_540656)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_540658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540658)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_540658


    @norecursion
    def test_smallx(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smallx'
        module_type_store = module_type_store.open_function_context('test_smallx', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_localization', localization)
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_function_name', 'TestKolmogorov.test_smallx')
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_param_names_list', [])
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKolmogorov.test_smallx.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogorov.test_smallx', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smallx', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smallx(...)' code ##################

        
        # Assigning a BinOp to a Name (line 192):
        float_540659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 18), 'float')
        
        # Call to arange(...): (line 192)
        # Processing the call arguments (line 192)
        int_540662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 35), 'int')
        int_540663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 38), 'int')
        # Processing the call keyword arguments (line 192)
        kwargs_540664 = {}
        # Getting the type of 'np' (line 192)
        np_540660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 'np', False)
        # Obtaining the member 'arange' of a type (line 192)
        arange_540661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 25), np_540660, 'arange')
        # Calling arange(args, kwargs) (line 192)
        arange_call_result_540665 = invoke(stypy.reporting.localization.Localization(__file__, 192, 25), arange_540661, *[int_540662, int_540663], **kwargs_540664)
        
        # Applying the binary operator '**' (line 192)
        result_pow_540666 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 18), '**', float_540659, arange_call_result_540665)
        
        # Assigning a type to the variable 'epsilon' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'epsilon', result_pow_540666)
        
        # Assigning a Call to a Name (line 193):
        
        # Call to array(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Obtaining an instance of the builtin type 'list' (line 193)
        list_540669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 193)
        # Adding element type (line 193)
        float_540670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540670)
        # Adding element type (line 193)
        float_540671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540671)
        # Adding element type (line 193)
        float_540672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540672)
        # Adding element type (line 193)
        float_540673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540673)
        # Adding element type (line 193)
        float_540674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540674)
        # Adding element type (line 193)
        float_540675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540675)
        # Adding element type (line 193)
        float_540676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540676)
        # Adding element type (line 193)
        float_540677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540677)
        # Adding element type (line 193)
        float_540678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540678)
        # Adding element type (line 193)
        float_540679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540679)
        # Adding element type (line 193)
        float_540680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540680)
        # Adding element type (line 193)
        float_540681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540681)
        # Adding element type (line 193)
        float_540682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), list_540669, float_540682)
        
        # Processing the call keyword arguments (line 193)
        kwargs_540683 = {}
        # Getting the type of 'np' (line 193)
        np_540667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 193)
        array_540668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), np_540667, 'array')
        # Calling array(args, kwargs) (line 193)
        array_call_result_540684 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), array_540668, *[list_540669], **kwargs_540683)
        
        # Assigning a type to the variable 'x' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'x', array_call_result_540684)
        
        # Assigning a Call to a Name (line 198):
        
        # Call to column_stack(...): (line 198)
        # Processing the call arguments (line 198)
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_540687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        # Adding element type (line 198)
        # Getting the type of 'x' (line 198)
        x_540688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 35), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 34), list_540687, x_540688)
        # Adding element type (line 198)
        int_540689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 38), 'int')
        # Getting the type of 'epsilon' (line 198)
        epsilon_540690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 40), 'epsilon', False)
        # Applying the binary operator '-' (line 198)
        result_sub_540691 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 38), '-', int_540689, epsilon_540690)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 34), list_540687, result_sub_540691)
        
        # Processing the call keyword arguments (line 198)
        kwargs_540692 = {}
        # Getting the type of 'np' (line 198)
        np_540685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 198)
        column_stack_540686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 18), np_540685, 'column_stack')
        # Calling column_stack(args, kwargs) (line 198)
        column_stack_call_result_540693 = invoke(stypy.reporting.localization.Localization(__file__, 198, 18), column_stack_540686, *[list_540687], **kwargs_540692)
        
        # Assigning a type to the variable 'dataset' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'dataset', column_stack_call_result_540693)
        
        # Call to check(...): (line 199)
        # Processing the call keyword arguments (line 199)
        kwargs_540705 = {}
        
        # Call to FuncData(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'kolmogorov' (line 199)
        kolmogorov_540695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 17), 'kolmogorov', False)
        # Getting the type of 'dataset' (line 199)
        dataset_540696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 199)
        tuple_540697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 199)
        # Adding element type (line 199)
        int_540698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 39), tuple_540697, int_540698)
        
        int_540699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 44), 'int')
        # Processing the call keyword arguments (line 199)
        # Getting the type of '_rtol' (line 199)
        _rtol_540700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 52), '_rtol', False)
        keyword_540701 = _rtol_540700
        kwargs_540702 = {'rtol': keyword_540701}
        # Getting the type of 'FuncData' (line 199)
        FuncData_540694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 199)
        FuncData_call_result_540703 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), FuncData_540694, *[kolmogorov_540695, dataset_540696, tuple_540697, int_540699], **kwargs_540702)
        
        # Obtaining the member 'check' of a type (line 199)
        check_540704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), FuncData_call_result_540703, 'check')
        # Calling check(args, kwargs) (line 199)
        check_call_result_540706 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), check_540704, *[], **kwargs_540705)
        
        
        # ################# End of 'test_smallx(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smallx' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_540707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540707)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smallx'
        return stypy_return_type_540707


    @norecursion
    def test_round_trip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_round_trip'
        module_type_store = module_type_store.open_function_context('test_round_trip', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_localization', localization)
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_function_name', 'TestKolmogorov.test_round_trip')
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_param_names_list', [])
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKolmogorov.test_round_trip.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogorov.test_round_trip', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_round_trip', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_round_trip(...)' code ##################


        @norecursion
        def _ki_k(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_ki_k'
            module_type_store = module_type_store.open_function_context('_ki_k', 203, 8, False)
            
            # Passed parameters checking function
            _ki_k.stypy_localization = localization
            _ki_k.stypy_type_of_self = None
            _ki_k.stypy_type_store = module_type_store
            _ki_k.stypy_function_name = '_ki_k'
            _ki_k.stypy_param_names_list = ['_x']
            _ki_k.stypy_varargs_param_name = None
            _ki_k.stypy_kwargs_param_name = None
            _ki_k.stypy_call_defaults = defaults
            _ki_k.stypy_call_varargs = varargs
            _ki_k.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_ki_k', ['_x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_ki_k', localization, ['_x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_ki_k(...)' code ##################

            
            # Call to kolmogi(...): (line 204)
            # Processing the call arguments (line 204)
            
            # Call to kolmogorov(...): (line 204)
            # Processing the call arguments (line 204)
            # Getting the type of '_x' (line 204)
            _x_540710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 38), '_x', False)
            # Processing the call keyword arguments (line 204)
            kwargs_540711 = {}
            # Getting the type of 'kolmogorov' (line 204)
            kolmogorov_540709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 27), 'kolmogorov', False)
            # Calling kolmogorov(args, kwargs) (line 204)
            kolmogorov_call_result_540712 = invoke(stypy.reporting.localization.Localization(__file__, 204, 27), kolmogorov_540709, *[_x_540710], **kwargs_540711)
            
            # Processing the call keyword arguments (line 204)
            kwargs_540713 = {}
            # Getting the type of 'kolmogi' (line 204)
            kolmogi_540708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'kolmogi', False)
            # Calling kolmogi(args, kwargs) (line 204)
            kolmogi_call_result_540714 = invoke(stypy.reporting.localization.Localization(__file__, 204, 19), kolmogi_540708, *[kolmogorov_call_result_540712], **kwargs_540713)
            
            # Assigning a type to the variable 'stypy_return_type' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'stypy_return_type', kolmogi_call_result_540714)
            
            # ################# End of '_ki_k(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_ki_k' in the type store
            # Getting the type of 'stypy_return_type' (line 203)
            stypy_return_type_540715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_540715)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_ki_k'
            return stypy_return_type_540715

        # Assigning a type to the variable '_ki_k' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), '_ki_k', _ki_k)
        
        # Assigning a Call to a Name (line 206):
        
        # Call to linspace(...): (line 206)
        # Processing the call arguments (line 206)
        float_540718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 24), 'float')
        float_540719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 29), 'float')
        int_540720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 34), 'int')
        # Processing the call keyword arguments (line 206)
        # Getting the type of 'True' (line 206)
        True_540721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 47), 'True', False)
        keyword_540722 = True_540721
        kwargs_540723 = {'endpoint': keyword_540722}
        # Getting the type of 'np' (line 206)
        np_540716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 206)
        linspace_540717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), np_540716, 'linspace')
        # Calling linspace(args, kwargs) (line 206)
        linspace_call_result_540724 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), linspace_540717, *[float_540718, float_540719, int_540720], **kwargs_540723)
        
        # Assigning a type to the variable 'x' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'x', linspace_call_result_540724)
        
        # Assigning a Call to a Name (line 207):
        
        # Call to column_stack(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_540727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        # Getting the type of 'x' (line 207)
        x_540728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 35), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 34), list_540727, x_540728)
        # Adding element type (line 207)
        # Getting the type of 'x' (line 207)
        x_540729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 34), list_540727, x_540729)
        
        # Processing the call keyword arguments (line 207)
        kwargs_540730 = {}
        # Getting the type of 'np' (line 207)
        np_540725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 207)
        column_stack_540726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 18), np_540725, 'column_stack')
        # Calling column_stack(args, kwargs) (line 207)
        column_stack_call_result_540731 = invoke(stypy.reporting.localization.Localization(__file__, 207, 18), column_stack_540726, *[list_540727], **kwargs_540730)
        
        # Assigning a type to the variable 'dataset' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'dataset', column_stack_call_result_540731)
        
        # Call to check(...): (line 208)
        # Processing the call keyword arguments (line 208)
        kwargs_540743 = {}
        
        # Call to FuncData(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of '_ki_k' (line 208)
        _ki_k_540733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), '_ki_k', False)
        # Getting the type of 'dataset' (line 208)
        dataset_540734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 24), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 208)
        tuple_540735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 208)
        # Adding element type (line 208)
        int_540736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 34), tuple_540735, int_540736)
        
        int_540737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 39), 'int')
        # Processing the call keyword arguments (line 208)
        # Getting the type of '_rtol' (line 208)
        _rtol_540738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), '_rtol', False)
        keyword_540739 = _rtol_540738
        kwargs_540740 = {'rtol': keyword_540739}
        # Getting the type of 'FuncData' (line 208)
        FuncData_540732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 208)
        FuncData_call_result_540741 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), FuncData_540732, *[_ki_k_540733, dataset_540734, tuple_540735, int_540737], **kwargs_540740)
        
        # Obtaining the member 'check' of a type (line 208)
        check_540742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), FuncData_call_result_540741, 'check')
        # Calling check(args, kwargs) (line 208)
        check_call_result_540744 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), check_540742, *[], **kwargs_540743)
        
        
        # ################# End of 'test_round_trip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_round_trip' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_540745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_round_trip'
        return stypy_return_type_540745


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 178, 0, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogorov.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestKolmogorov' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'TestKolmogorov', TestKolmogorov)
# Declaration of the 'TestKolmogi' class

class TestKolmogi(object, ):

    @norecursion
    def test_nan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nan'
        module_type_store = module_type_store.open_function_context('test_nan', 212, 4, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_localization', localization)
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_function_name', 'TestKolmogi.test_nan')
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_param_names_list', [])
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKolmogi.test_nan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogi.test_nan', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nan(...)' code ##################

        
        # Call to assert_(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to isnan(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to kolmogi(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'np' (line 213)
        np_540750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'np', False)
        # Obtaining the member 'nan' of a type (line 213)
        nan_540751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 33), np_540750, 'nan')
        # Processing the call keyword arguments (line 213)
        kwargs_540752 = {}
        # Getting the type of 'kolmogi' (line 213)
        kolmogi_540749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'kolmogi', False)
        # Calling kolmogi(args, kwargs) (line 213)
        kolmogi_call_result_540753 = invoke(stypy.reporting.localization.Localization(__file__, 213, 25), kolmogi_540749, *[nan_540751], **kwargs_540752)
        
        # Processing the call keyword arguments (line 213)
        kwargs_540754 = {}
        # Getting the type of 'np' (line 213)
        np_540747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'np', False)
        # Obtaining the member 'isnan' of a type (line 213)
        isnan_540748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), np_540747, 'isnan')
        # Calling isnan(args, kwargs) (line 213)
        isnan_call_result_540755 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), isnan_540748, *[kolmogi_call_result_540753], **kwargs_540754)
        
        # Processing the call keyword arguments (line 213)
        kwargs_540756 = {}
        # Getting the type of 'assert_' (line 213)
        assert__540746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 213)
        assert__call_result_540757 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assert__540746, *[isnan_call_result_540755], **kwargs_540756)
        
        
        # ################# End of 'test_nan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nan' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_540758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nan'
        return stypy_return_type_540758


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_function_name', 'TestKolmogi.test_basic')
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKolmogi.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogi.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a List to a Name (line 217):
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_540759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_540760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        float_540761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 20), tuple_540760, float_540761)
        # Adding element type (line 217)
        int_540762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 20), tuple_540760, int_540762)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 18), list_540759, tuple_540760)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'tuple' (line 218)
        tuple_540763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 218)
        # Adding element type (line 218)
        float_540764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 20), tuple_540763, float_540764)
        # Adding element type (line 218)
        float_540765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 20), tuple_540763, float_540765)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 18), list_540759, tuple_540763)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'tuple' (line 219)
        tuple_540766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 219)
        # Adding element type (line 219)
        float_540767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 20), tuple_540766, float_540767)
        # Adding element type (line 219)
        int_540768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 20), tuple_540766, int_540768)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 18), list_540759, tuple_540766)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'tuple' (line 220)
        tuple_540769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 220)
        # Adding element type (line 220)
        float_540770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 20), tuple_540769, float_540770)
        # Adding element type (line 220)
        int_540771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 20), tuple_540769, int_540771)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 18), list_540759, tuple_540769)
        
        # Assigning a type to the variable 'dataset' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'dataset', list_540759)
        
        # Assigning a Call to a Name (line 222):
        
        # Call to asarray(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'dataset' (line 222)
        dataset_540774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 29), 'dataset', False)
        # Processing the call keyword arguments (line 222)
        kwargs_540775 = {}
        # Getting the type of 'np' (line 222)
        np_540772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 222)
        asarray_540773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 18), np_540772, 'asarray')
        # Calling asarray(args, kwargs) (line 222)
        asarray_call_result_540776 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), asarray_540773, *[dataset_540774], **kwargs_540775)
        
        # Assigning a type to the variable 'dataset' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'dataset', asarray_call_result_540776)
        
        # Call to check(...): (line 223)
        # Processing the call keyword arguments (line 223)
        kwargs_540788 = {}
        
        # Call to FuncData(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'kolmogi' (line 223)
        kolmogi_540778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'kolmogi', False)
        # Getting the type of 'dataset' (line 223)
        dataset_540779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 223)
        tuple_540780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 223)
        # Adding element type (line 223)
        int_540781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 36), tuple_540780, int_540781)
        
        int_540782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 41), 'int')
        # Processing the call keyword arguments (line 223)
        # Getting the type of '_rtol' (line 223)
        _rtol_540783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 49), '_rtol', False)
        keyword_540784 = _rtol_540783
        kwargs_540785 = {'rtol': keyword_540784}
        # Getting the type of 'FuncData' (line 223)
        FuncData_540777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 223)
        FuncData_call_result_540786 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), FuncData_540777, *[kolmogi_540778, dataset_540779, tuple_540780, int_540782], **kwargs_540785)
        
        # Obtaining the member 'check' of a type (line 223)
        check_540787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), FuncData_call_result_540786, 'check')
        # Calling check(args, kwargs) (line 223)
        check_call_result_540789 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), check_540787, *[], **kwargs_540788)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_540790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540790)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_540790


    @norecursion
    def test_smallp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smallp'
        module_type_store = module_type_store.open_function_context('test_smallp', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_localization', localization)
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_function_name', 'TestKolmogi.test_smallp')
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_param_names_list', [])
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKolmogi.test_smallp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogi.test_smallp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smallp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smallp(...)' code ##################

        
        # Assigning a BinOp to a Name (line 227):
        float_540791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 18), 'float')
        
        # Call to arange(...): (line 227)
        # Processing the call arguments (line 227)
        int_540794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 35), 'int')
        int_540795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 38), 'int')
        # Processing the call keyword arguments (line 227)
        kwargs_540796 = {}
        # Getting the type of 'np' (line 227)
        np_540792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'np', False)
        # Obtaining the member 'arange' of a type (line 227)
        arange_540793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 25), np_540792, 'arange')
        # Calling arange(args, kwargs) (line 227)
        arange_call_result_540797 = invoke(stypy.reporting.localization.Localization(__file__, 227, 25), arange_540793, *[int_540794, int_540795], **kwargs_540796)
        
        # Applying the binary operator '**' (line 227)
        result_pow_540798 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 18), '**', float_540791, arange_call_result_540797)
        
        # Assigning a type to the variable 'epsilon' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'epsilon', result_pow_540798)
        
        # Assigning a Call to a Name (line 228):
        
        # Call to array(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Obtaining an instance of the builtin type 'list' (line 228)
        list_540801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 228)
        # Adding element type (line 228)
        float_540802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540802)
        # Adding element type (line 228)
        float_540803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540803)
        # Adding element type (line 228)
        float_540804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540804)
        # Adding element type (line 228)
        float_540805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540805)
        # Adding element type (line 228)
        float_540806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540806)
        # Adding element type (line 228)
        float_540807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540807)
        # Adding element type (line 228)
        float_540808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540808)
        # Adding element type (line 228)
        float_540809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540809)
        # Adding element type (line 228)
        float_540810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540810)
        # Adding element type (line 228)
        float_540811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540811)
        # Adding element type (line 228)
        float_540812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540812)
        # Adding element type (line 228)
        float_540813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540813)
        # Adding element type (line 228)
        float_540814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 21), list_540801, float_540814)
        
        # Processing the call keyword arguments (line 228)
        kwargs_540815 = {}
        # Getting the type of 'np' (line 228)
        np_540799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 228)
        array_540800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), np_540799, 'array')
        # Calling array(args, kwargs) (line 228)
        array_call_result_540816 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), array_540800, *[list_540801], **kwargs_540815)
        
        # Assigning a type to the variable 'x' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'x', array_call_result_540816)
        
        # Assigning a Call to a Name (line 233):
        
        # Call to column_stack(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_540819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        # Adding element type (line 233)
        int_540820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 35), 'int')
        # Getting the type of 'epsilon' (line 233)
        epsilon_540821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 37), 'epsilon', False)
        # Applying the binary operator '-' (line 233)
        result_sub_540822 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 35), '-', int_540820, epsilon_540821)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 34), list_540819, result_sub_540822)
        # Adding element type (line 233)
        # Getting the type of 'x' (line 233)
        x_540823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 46), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 34), list_540819, x_540823)
        
        # Processing the call keyword arguments (line 233)
        kwargs_540824 = {}
        # Getting the type of 'np' (line 233)
        np_540817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 233)
        column_stack_540818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 18), np_540817, 'column_stack')
        # Calling column_stack(args, kwargs) (line 233)
        column_stack_call_result_540825 = invoke(stypy.reporting.localization.Localization(__file__, 233, 18), column_stack_540818, *[list_540819], **kwargs_540824)
        
        # Assigning a type to the variable 'dataset' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'dataset', column_stack_call_result_540825)
        
        # Call to check(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_540837 = {}
        
        # Call to FuncData(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'kolmogi' (line 234)
        kolmogi_540827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'kolmogi', False)
        # Getting the type of 'dataset' (line 234)
        dataset_540828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 26), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 234)
        tuple_540829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 234)
        # Adding element type (line 234)
        int_540830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 36), tuple_540829, int_540830)
        
        int_540831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 41), 'int')
        # Processing the call keyword arguments (line 234)
        # Getting the type of '_rtol' (line 234)
        _rtol_540832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 49), '_rtol', False)
        keyword_540833 = _rtol_540832
        kwargs_540834 = {'rtol': keyword_540833}
        # Getting the type of 'FuncData' (line 234)
        FuncData_540826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 234)
        FuncData_call_result_540835 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), FuncData_540826, *[kolmogi_540827, dataset_540828, tuple_540829, int_540831], **kwargs_540834)
        
        # Obtaining the member 'check' of a type (line 234)
        check_540836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), FuncData_call_result_540835, 'check')
        # Calling check(args, kwargs) (line 234)
        check_call_result_540838 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), check_540836, *[], **kwargs_540837)
        
        
        # ################# End of 'test_smallp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smallp' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_540839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540839)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smallp'
        return stypy_return_type_540839


    @norecursion
    def test_round_trip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_round_trip'
        module_type_store = module_type_store.open_function_context('test_round_trip', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_localization', localization)
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_function_name', 'TestKolmogi.test_round_trip')
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_param_names_list', [])
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKolmogi.test_round_trip.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogi.test_round_trip', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_round_trip', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_round_trip(...)' code ##################


        @norecursion
        def _k_ki(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_k_ki'
            module_type_store = module_type_store.open_function_context('_k_ki', 237, 8, False)
            
            # Passed parameters checking function
            _k_ki.stypy_localization = localization
            _k_ki.stypy_type_of_self = None
            _k_ki.stypy_type_store = module_type_store
            _k_ki.stypy_function_name = '_k_ki'
            _k_ki.stypy_param_names_list = ['_p']
            _k_ki.stypy_varargs_param_name = None
            _k_ki.stypy_kwargs_param_name = None
            _k_ki.stypy_call_defaults = defaults
            _k_ki.stypy_call_varargs = varargs
            _k_ki.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_k_ki', ['_p'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_k_ki', localization, ['_p'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_k_ki(...)' code ##################

            
            # Call to kolmogorov(...): (line 238)
            # Processing the call arguments (line 238)
            
            # Call to kolmogi(...): (line 238)
            # Processing the call arguments (line 238)
            # Getting the type of '_p' (line 238)
            _p_540842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 38), '_p', False)
            # Processing the call keyword arguments (line 238)
            kwargs_540843 = {}
            # Getting the type of 'kolmogi' (line 238)
            kolmogi_540841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 30), 'kolmogi', False)
            # Calling kolmogi(args, kwargs) (line 238)
            kolmogi_call_result_540844 = invoke(stypy.reporting.localization.Localization(__file__, 238, 30), kolmogi_540841, *[_p_540842], **kwargs_540843)
            
            # Processing the call keyword arguments (line 238)
            kwargs_540845 = {}
            # Getting the type of 'kolmogorov' (line 238)
            kolmogorov_540840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 19), 'kolmogorov', False)
            # Calling kolmogorov(args, kwargs) (line 238)
            kolmogorov_call_result_540846 = invoke(stypy.reporting.localization.Localization(__file__, 238, 19), kolmogorov_540840, *[kolmogi_call_result_540844], **kwargs_540845)
            
            # Assigning a type to the variable 'stypy_return_type' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'stypy_return_type', kolmogorov_call_result_540846)
            
            # ################# End of '_k_ki(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_k_ki' in the type store
            # Getting the type of 'stypy_return_type' (line 237)
            stypy_return_type_540847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_540847)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_k_ki'
            return stypy_return_type_540847

        # Assigning a type to the variable '_k_ki' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), '_k_ki', _k_ki)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to linspace(...): (line 240)
        # Processing the call arguments (line 240)
        float_540850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 24), 'float')
        float_540851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 29), 'float')
        int_540852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 34), 'int')
        # Processing the call keyword arguments (line 240)
        # Getting the type of 'True' (line 240)
        True_540853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 47), 'True', False)
        keyword_540854 = True_540853
        kwargs_540855 = {'endpoint': keyword_540854}
        # Getting the type of 'np' (line 240)
        np_540848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 240)
        linspace_540849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), np_540848, 'linspace')
        # Calling linspace(args, kwargs) (line 240)
        linspace_call_result_540856 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), linspace_540849, *[float_540850, float_540851, int_540852], **kwargs_540855)
        
        # Assigning a type to the variable 'p' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'p', linspace_call_result_540856)
        
        # Assigning a Call to a Name (line 241):
        
        # Call to column_stack(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_540859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        # Getting the type of 'p' (line 241)
        p_540860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 35), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 34), list_540859, p_540860)
        # Adding element type (line 241)
        # Getting the type of 'p' (line 241)
        p_540861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 38), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 34), list_540859, p_540861)
        
        # Processing the call keyword arguments (line 241)
        kwargs_540862 = {}
        # Getting the type of 'np' (line 241)
        np_540857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 241)
        column_stack_540858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 18), np_540857, 'column_stack')
        # Calling column_stack(args, kwargs) (line 241)
        column_stack_call_result_540863 = invoke(stypy.reporting.localization.Localization(__file__, 241, 18), column_stack_540858, *[list_540859], **kwargs_540862)
        
        # Assigning a type to the variable 'dataset' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'dataset', column_stack_call_result_540863)
        
        # Call to check(...): (line 242)
        # Processing the call keyword arguments (line 242)
        kwargs_540875 = {}
        
        # Call to FuncData(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of '_k_ki' (line 242)
        _k_ki_540865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 17), '_k_ki', False)
        # Getting the type of 'dataset' (line 242)
        dataset_540866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'dataset', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 242)
        tuple_540867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 242)
        # Adding element type (line 242)
        int_540868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 34), tuple_540867, int_540868)
        
        int_540869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 39), 'int')
        # Processing the call keyword arguments (line 242)
        # Getting the type of '_rtol' (line 242)
        _rtol_540870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 47), '_rtol', False)
        keyword_540871 = _rtol_540870
        kwargs_540872 = {'rtol': keyword_540871}
        # Getting the type of 'FuncData' (line 242)
        FuncData_540864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 242)
        FuncData_call_result_540873 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), FuncData_540864, *[_k_ki_540865, dataset_540866, tuple_540867, int_540869], **kwargs_540872)
        
        # Obtaining the member 'check' of a type (line 242)
        check_540874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), FuncData_call_result_540873, 'check')
        # Calling check(args, kwargs) (line 242)
        check_call_result_540876 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), check_540874, *[], **kwargs_540875)
        
        
        # ################# End of 'test_round_trip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_round_trip' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_540877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_540877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_round_trip'
        return stypy_return_type_540877


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 211, 0, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKolmogi.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestKolmogi' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'TestKolmogi', TestKolmogi)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
