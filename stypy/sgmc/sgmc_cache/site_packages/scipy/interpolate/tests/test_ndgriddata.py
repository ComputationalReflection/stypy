
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_equal, assert_array_equal, assert_allclose
5: from pytest import raises as assert_raises
6: 
7: from scipy.interpolate import griddata, NearestNDInterpolator
8: 
9: 
10: class TestGriddata(object):
11:     def test_fill_value(self):
12:         x = [(0,0), (0,1), (1,0)]
13:         y = [1, 2, 3]
14: 
15:         yi = griddata(x, y, [(1,1), (1,2), (0,0)], fill_value=-1)
16:         assert_array_equal(yi, [-1., -1, 1])
17: 
18:         yi = griddata(x, y, [(1,1), (1,2), (0,0)])
19:         assert_array_equal(yi, [np.nan, np.nan, 1])
20: 
21:     def test_alternative_call(self):
22:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
23:                      dtype=np.double)
24:         y = (np.arange(x.shape[0], dtype=np.double)[:,None]
25:              + np.array([0,1])[None,:])
26: 
27:         for method in ('nearest', 'linear', 'cubic'):
28:             for rescale in (True, False):
29:                 msg = repr((method, rescale))
30:                 yi = griddata((x[:,0], x[:,1]), y, (x[:,0], x[:,1]), method=method,
31:                               rescale=rescale)
32:                 assert_allclose(y, yi, atol=1e-14, err_msg=msg)
33: 
34:     def test_multivalue_2d(self):
35:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
36:                      dtype=np.double)
37:         y = (np.arange(x.shape[0], dtype=np.double)[:,None]
38:              + np.array([0,1])[None,:])
39: 
40:         for method in ('nearest', 'linear', 'cubic'):
41:             for rescale in (True, False):
42:                 msg = repr((method, rescale))
43:                 yi = griddata(x, y, x, method=method, rescale=rescale)
44:                 assert_allclose(y, yi, atol=1e-14, err_msg=msg)
45: 
46:     def test_multipoint_2d(self):
47:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
48:                      dtype=np.double)
49:         y = np.arange(x.shape[0], dtype=np.double)
50: 
51:         xi = x[:,None,:] + np.array([0,0,0])[None,:,None]
52: 
53:         for method in ('nearest', 'linear', 'cubic'):
54:             for rescale in (True, False):
55:                 msg = repr((method, rescale))
56:                 yi = griddata(x, y, xi, method=method, rescale=rescale)
57: 
58:                 assert_equal(yi.shape, (5, 3), err_msg=msg)
59:                 assert_allclose(yi, np.tile(y[:,None], (1, 3)),
60:                                 atol=1e-14, err_msg=msg)
61: 
62:     def test_complex_2d(self):
63:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
64:                      dtype=np.double)
65:         y = np.arange(x.shape[0], dtype=np.double)
66:         y = y - 2j*y[::-1]
67: 
68:         xi = x[:,None,:] + np.array([0,0,0])[None,:,None]
69: 
70:         for method in ('nearest', 'linear', 'cubic'):
71:             for rescale in (True, False):
72:                 msg = repr((method, rescale))
73:                 yi = griddata(x, y, xi, method=method, rescale=rescale)
74: 
75:                 assert_equal(yi.shape, (5, 3), err_msg=msg)
76:                 assert_allclose(yi, np.tile(y[:,None], (1, 3)),
77:                                 atol=1e-14, err_msg=msg)
78: 
79:     def test_1d(self):
80:         x = np.array([1, 2.5, 3, 4.5, 5, 6])
81:         y = np.array([1, 2, 0, 3.9, 2, 1])
82: 
83:         for method in ('nearest', 'linear', 'cubic'):
84:             assert_allclose(griddata(x, y, x, method=method), y,
85:                             err_msg=method, atol=1e-14)
86:             assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y,
87:                             err_msg=method, atol=1e-14)
88:             assert_allclose(griddata((x,), y, (x,), method=method), y,
89:                             err_msg=method, atol=1e-14)
90: 
91:     def test_1d_borders(self):
92:         # Test for nearest neighbor case with xi outside
93:         # the range of the values.
94:         x = np.array([1, 2.5, 3, 4.5, 5, 6])
95:         y = np.array([1, 2, 0, 3.9, 2, 1])
96:         xi = np.array([0.9, 6.5])
97:         yi_should = np.array([1.0, 1.0])
98: 
99:         method = 'nearest'
100:         assert_allclose(griddata(x, y, xi,
101:                                  method=method), yi_should,
102:                         err_msg=method,
103:                         atol=1e-14)
104:         assert_allclose(griddata(x.reshape(6, 1), y, xi,
105:                                  method=method), yi_should,
106:                         err_msg=method,
107:                         atol=1e-14)
108:         assert_allclose(griddata((x, ), y, (xi, ),
109:                                  method=method), yi_should,
110:                         err_msg=method,
111:                         atol=1e-14)
112: 
113:     def test_1d_unsorted(self):
114:         x = np.array([2.5, 1, 4.5, 5, 6, 3])
115:         y = np.array([1, 2, 0, 3.9, 2, 1])
116: 
117:         for method in ('nearest', 'linear', 'cubic'):
118:             assert_allclose(griddata(x, y, x, method=method), y,
119:                             err_msg=method, atol=1e-10)
120:             assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y,
121:                             err_msg=method, atol=1e-10)
122:             assert_allclose(griddata((x,), y, (x,), method=method), y,
123:                             err_msg=method, atol=1e-10)
124: 
125:     def test_square_rescale_manual(self):
126:         points = np.array([(0,0), (0,100), (10,100), (10,0), (1, 5)], dtype=np.double)
127:         points_rescaled = np.array([(0,0), (0,1), (1,1), (1,0), (0.1, 0.05)], dtype=np.double)
128:         values = np.array([1., 2., -3., 5., 9.], dtype=np.double)
129: 
130:         xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:,None],
131:                                      np.linspace(0, 100, 14)[None,:])
132:         xx = xx.ravel()
133:         yy = yy.ravel()
134:         xi = np.array([xx, yy]).T.copy()
135: 
136:         for method in ('nearest', 'linear', 'cubic'):
137:             msg = method
138:             zi = griddata(points_rescaled, values, xi/np.array([10, 100.]),
139:                           method=method)
140:             zi_rescaled = griddata(points, values, xi, method=method,
141:                                    rescale=True)
142:             assert_allclose(zi, zi_rescaled, err_msg=msg,
143:                             atol=1e-12)
144: 
145:     def test_xi_1d(self):
146:         # Check that 1-D xi is interpreted as a coordinate
147:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
148:                      dtype=np.double)
149:         y = np.arange(x.shape[0], dtype=np.double)
150:         y = y - 2j*y[::-1]
151: 
152:         xi = np.array([0.5, 0.5])
153: 
154:         for method in ('nearest', 'linear', 'cubic'):
155:             p1 = griddata(x, y, xi, method=method)
156:             p2 = griddata(x, y, xi[None,:], method=method)
157:             assert_allclose(p1, p2, err_msg=method)
158: 
159:             xi1 = np.array([0.5])
160:             xi3 = np.array([0.5, 0.5, 0.5])
161:             assert_raises(ValueError, griddata, x, y, xi1,
162:                           method=method)
163:             assert_raises(ValueError, griddata, x, y, xi3,
164:                           method=method)
165:         
166: 
167: def test_nearest_options():
168:     # smoke test that NearestNDInterpolator accept cKDTree options
169:     npts, nd = 4, 3
170:     x = np.arange(npts*nd).reshape((npts, nd))
171:     y = np.arange(npts)
172:     nndi = NearestNDInterpolator(x, y)
173: 
174:     opts = {'balanced_tree': False, 'compact_nodes': False}
175:     nndi_o = NearestNDInterpolator(x, y, tree_options=opts)
176:     assert_allclose(nndi(x), nndi_o(x), atol=1e-14)
177: 
178: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113911 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_113911) is not StypyTypeError):

    if (import_113911 != 'pyd_module'):
        __import__(import_113911)
        sys_modules_113912 = sys.modules[import_113911]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_113912.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_113911)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_array_equal, assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113913 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_113913) is not StypyTypeError):

    if (import_113913 != 'pyd_module'):
        __import__(import_113913)
        sys_modules_113914 = sys.modules[import_113913]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_113914.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_113914, sys_modules_113914.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_allclose'], [assert_equal, assert_array_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_113913)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from pytest import assert_raises' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113915 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_113915) is not StypyTypeError):

    if (import_113915 != 'pyd_module'):
        __import__(import_113915)
        sys_modules_113916 = sys.modules[import_113915]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_113916.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_113916, sys_modules_113916.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_113915)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.interpolate import griddata, NearestNDInterpolator' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113917 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.interpolate')

if (type(import_113917) is not StypyTypeError):

    if (import_113917 != 'pyd_module'):
        __import__(import_113917)
        sys_modules_113918 = sys.modules[import_113917]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.interpolate', sys_modules_113918.module_type_store, module_type_store, ['griddata', 'NearestNDInterpolator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_113918, sys_modules_113918.module_type_store, module_type_store)
    else:
        from scipy.interpolate import griddata, NearestNDInterpolator

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.interpolate', None, module_type_store, ['griddata', 'NearestNDInterpolator'], [griddata, NearestNDInterpolator])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.interpolate', import_113917)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

# Declaration of the 'TestGriddata' class

class TestGriddata(object, ):

    @norecursion
    def test_fill_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fill_value'
        module_type_store = module_type_store.open_function_context('test_fill_value', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_fill_value')
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_fill_value.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_fill_value', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fill_value', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fill_value(...)' code ##################

        
        # Assigning a List to a Name (line 12):
        
        # Assigning a List to a Name (line 12):
        
        # Obtaining an instance of the builtin type 'list' (line 12)
        list_113919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 12)
        # Adding element type (line 12)
        
        # Obtaining an instance of the builtin type 'tuple' (line 12)
        tuple_113920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 12)
        # Adding element type (line 12)
        int_113921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), tuple_113920, int_113921)
        # Adding element type (line 12)
        int_113922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), tuple_113920, int_113922)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 12), list_113919, tuple_113920)
        # Adding element type (line 12)
        
        # Obtaining an instance of the builtin type 'tuple' (line 12)
        tuple_113923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 12)
        # Adding element type (line 12)
        int_113924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), tuple_113923, int_113924)
        # Adding element type (line 12)
        int_113925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), tuple_113923, int_113925)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 12), list_113919, tuple_113923)
        # Adding element type (line 12)
        
        # Obtaining an instance of the builtin type 'tuple' (line 12)
        tuple_113926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 12)
        # Adding element type (line 12)
        int_113927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 28), tuple_113926, int_113927)
        # Adding element type (line 12)
        int_113928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 28), tuple_113926, int_113928)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 12), list_113919, tuple_113926)
        
        # Assigning a type to the variable 'x' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'x', list_113919)
        
        # Assigning a List to a Name (line 13):
        
        # Assigning a List to a Name (line 13):
        
        # Obtaining an instance of the builtin type 'list' (line 13)
        list_113929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 13)
        # Adding element type (line 13)
        int_113930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 12), list_113929, int_113930)
        # Adding element type (line 13)
        int_113931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 12), list_113929, int_113931)
        # Adding element type (line 13)
        int_113932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 12), list_113929, int_113932)
        
        # Assigning a type to the variable 'y' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'y', list_113929)
        
        # Assigning a Call to a Name (line 15):
        
        # Assigning a Call to a Name (line 15):
        
        # Call to griddata(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'x' (line 15)
        x_113934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'x', False)
        # Getting the type of 'y' (line 15)
        y_113935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_113936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        # Adding element type (line 15)
        
        # Obtaining an instance of the builtin type 'tuple' (line 15)
        tuple_113937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 15)
        # Adding element type (line 15)
        int_113938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 30), tuple_113937, int_113938)
        # Adding element type (line 15)
        int_113939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 30), tuple_113937, int_113939)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 28), list_113936, tuple_113937)
        # Adding element type (line 15)
        
        # Obtaining an instance of the builtin type 'tuple' (line 15)
        tuple_113940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 15)
        # Adding element type (line 15)
        int_113941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 37), tuple_113940, int_113941)
        # Adding element type (line 15)
        int_113942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 37), tuple_113940, int_113942)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 28), list_113936, tuple_113940)
        # Adding element type (line 15)
        
        # Obtaining an instance of the builtin type 'tuple' (line 15)
        tuple_113943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 15)
        # Adding element type (line 15)
        int_113944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 44), tuple_113943, int_113944)
        # Adding element type (line 15)
        int_113945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 44), tuple_113943, int_113945)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 28), list_113936, tuple_113943)
        
        # Processing the call keyword arguments (line 15)
        int_113946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 62), 'int')
        keyword_113947 = int_113946
        kwargs_113948 = {'fill_value': keyword_113947}
        # Getting the type of 'griddata' (line 15)
        griddata_113933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'griddata', False)
        # Calling griddata(args, kwargs) (line 15)
        griddata_call_result_113949 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), griddata_113933, *[x_113934, y_113935, list_113936], **kwargs_113948)
        
        # Assigning a type to the variable 'yi' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'yi', griddata_call_result_113949)
        
        # Call to assert_array_equal(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'yi' (line 16)
        yi_113951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'yi', False)
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_113952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        # Adding element type (line 16)
        float_113953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 31), list_113952, float_113953)
        # Adding element type (line 16)
        int_113954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 31), list_113952, int_113954)
        # Adding element type (line 16)
        int_113955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 31), list_113952, int_113955)
        
        # Processing the call keyword arguments (line 16)
        kwargs_113956 = {}
        # Getting the type of 'assert_array_equal' (line 16)
        assert_array_equal_113950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 16)
        assert_array_equal_call_result_113957 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), assert_array_equal_113950, *[yi_113951, list_113952], **kwargs_113956)
        
        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to griddata(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'x' (line 18)
        x_113959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'x', False)
        # Getting the type of 'y' (line 18)
        y_113960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_113961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'tuple' (line 18)
        tuple_113962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 18)
        # Adding element type (line 18)
        int_113963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 30), tuple_113962, int_113963)
        # Adding element type (line 18)
        int_113964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 30), tuple_113962, int_113964)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 28), list_113961, tuple_113962)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'tuple' (line 18)
        tuple_113965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 18)
        # Adding element type (line 18)
        int_113966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 37), tuple_113965, int_113966)
        # Adding element type (line 18)
        int_113967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 37), tuple_113965, int_113967)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 28), list_113961, tuple_113965)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'tuple' (line 18)
        tuple_113968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 18)
        # Adding element type (line 18)
        int_113969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 44), tuple_113968, int_113969)
        # Adding element type (line 18)
        int_113970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 44), tuple_113968, int_113970)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 28), list_113961, tuple_113968)
        
        # Processing the call keyword arguments (line 18)
        kwargs_113971 = {}
        # Getting the type of 'griddata' (line 18)
        griddata_113958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'griddata', False)
        # Calling griddata(args, kwargs) (line 18)
        griddata_call_result_113972 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), griddata_113958, *[x_113959, y_113960, list_113961], **kwargs_113971)
        
        # Assigning a type to the variable 'yi' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'yi', griddata_call_result_113972)
        
        # Call to assert_array_equal(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'yi' (line 19)
        yi_113974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'yi', False)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_113975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        # Getting the type of 'np' (line 19)
        np_113976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 32), 'np', False)
        # Obtaining the member 'nan' of a type (line 19)
        nan_113977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 32), np_113976, 'nan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 31), list_113975, nan_113977)
        # Adding element type (line 19)
        # Getting the type of 'np' (line 19)
        np_113978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 40), 'np', False)
        # Obtaining the member 'nan' of a type (line 19)
        nan_113979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 40), np_113978, 'nan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 31), list_113975, nan_113979)
        # Adding element type (line 19)
        int_113980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 31), list_113975, int_113980)
        
        # Processing the call keyword arguments (line 19)
        kwargs_113981 = {}
        # Getting the type of 'assert_array_equal' (line 19)
        assert_array_equal_113973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 19)
        assert_array_equal_call_result_113982 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), assert_array_equal_113973, *[yi_113974, list_113975], **kwargs_113981)
        
        
        # ################# End of 'test_fill_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fill_value' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_113983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113983)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fill_value'
        return stypy_return_type_113983


    @norecursion
    def test_alternative_call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_alternative_call'
        module_type_store = module_type_store.open_function_context('test_alternative_call', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_alternative_call')
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_alternative_call.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_alternative_call', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_alternative_call', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_alternative_call(...)' code ##################

        
        # Assigning a Call to a Name (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Call to array(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_113986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_113987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        int_113988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), tuple_113987, int_113988)
        # Adding element type (line 22)
        int_113989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), tuple_113987, int_113989)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), list_113986, tuple_113987)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_113990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        float_113991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 30), tuple_113990, float_113991)
        # Adding element type (line 22)
        float_113992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 30), tuple_113990, float_113992)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), list_113986, tuple_113990)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_113993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        float_113994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 43), tuple_113993, float_113994)
        # Adding element type (line 22)
        float_113995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 43), tuple_113993, float_113995)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), list_113986, tuple_113993)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_113996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        float_113997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 55), tuple_113996, float_113997)
        # Adding element type (line 22)
        float_113998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 55), tuple_113996, float_113998)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), list_113986, tuple_113996)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_113999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        float_114000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 67), tuple_113999, float_114000)
        # Adding element type (line 22)
        float_114001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 67), tuple_113999, float_114001)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), list_113986, tuple_113999)
        
        # Processing the call keyword arguments (line 22)
        # Getting the type of 'np' (line 23)
        np_114002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 23)
        double_114003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 27), np_114002, 'double')
        keyword_114004 = double_114003
        kwargs_114005 = {'dtype': keyword_114004}
        # Getting the type of 'np' (line 22)
        np_113984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 22)
        array_113985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), np_113984, 'array')
        # Calling array(args, kwargs) (line 22)
        array_call_result_114006 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), array_113985, *[list_113986], **kwargs_114005)
        
        # Assigning a type to the variable 'x' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'x', array_call_result_114006)
        
        # Assigning a BinOp to a Name (line 24):
        
        # Assigning a BinOp to a Name (line 24):
        
        # Obtaining the type of the subscript
        slice_114007 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 13), None, None, None)
        # Getting the type of 'None' (line 24)
        None_114008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 54), 'None')
        
        # Call to arange(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining the type of the subscript
        int_114011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'int')
        # Getting the type of 'x' (line 24)
        x_114012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'x', False)
        # Obtaining the member 'shape' of a type (line 24)
        shape_114013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 23), x_114012, 'shape')
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___114014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 23), shape_114013, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_114015 = invoke(stypy.reporting.localization.Localization(__file__, 24, 23), getitem___114014, int_114011)
        
        # Processing the call keyword arguments (line 24)
        # Getting the type of 'np' (line 24)
        np_114016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'np', False)
        # Obtaining the member 'double' of a type (line 24)
        double_114017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 41), np_114016, 'double')
        keyword_114018 = double_114017
        kwargs_114019 = {'dtype': keyword_114018}
        # Getting the type of 'np' (line 24)
        np_114009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'np', False)
        # Obtaining the member 'arange' of a type (line 24)
        arange_114010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 13), np_114009, 'arange')
        # Calling arange(args, kwargs) (line 24)
        arange_call_result_114020 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), arange_114010, *[subscript_call_result_114015], **kwargs_114019)
        
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___114021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 13), arange_call_result_114020, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_114022 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), getitem___114021, (slice_114007, None_114008))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 25)
        None_114023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'None')
        slice_114024 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 15), None, None, None)
        
        # Call to array(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_114027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_114028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_114027, int_114028)
        # Adding element type (line 25)
        int_114029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_114027, int_114029)
        
        # Processing the call keyword arguments (line 25)
        kwargs_114030 = {}
        # Getting the type of 'np' (line 25)
        np_114025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 25)
        array_114026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), np_114025, 'array')
        # Calling array(args, kwargs) (line 25)
        array_call_result_114031 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), array_114026, *[list_114027], **kwargs_114030)
        
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___114032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), array_call_result_114031, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_114033 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), getitem___114032, (None_114023, slice_114024))
        
        # Applying the binary operator '+' (line 24)
        result_add_114034 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 13), '+', subscript_call_result_114022, subscript_call_result_114033)
        
        # Assigning a type to the variable 'y' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'y', result_add_114034)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_114035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        str_114036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'str', 'nearest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), tuple_114035, str_114036)
        # Adding element type (line 27)
        str_114037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'str', 'linear')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), tuple_114035, str_114037)
        # Adding element type (line 27)
        str_114038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 44), 'str', 'cubic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), tuple_114035, str_114038)
        
        # Testing the type of a for loop iterable (line 27)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 8), tuple_114035)
        # Getting the type of the for loop variable (line 27)
        for_loop_var_114039 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 8), tuple_114035)
        # Assigning a type to the variable 'method' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'method', for_loop_var_114039)
        # SSA begins for a for statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_114040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'True' (line 28)
        True_114041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 28), tuple_114040, True_114041)
        # Adding element type (line 28)
        # Getting the type of 'False' (line 28)
        False_114042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 28), tuple_114040, False_114042)
        
        # Testing the type of a for loop iterable (line 28)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 28, 12), tuple_114040)
        # Getting the type of the for loop variable (line 28)
        for_loop_var_114043 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 28, 12), tuple_114040)
        # Assigning a type to the variable 'rescale' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'rescale', for_loop_var_114043)
        # SSA begins for a for statement (line 28)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Call to repr(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Obtaining an instance of the builtin type 'tuple' (line 29)
        tuple_114045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 29)
        # Adding element type (line 29)
        # Getting the type of 'method' (line 29)
        method_114046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'method', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 28), tuple_114045, method_114046)
        # Adding element type (line 29)
        # Getting the type of 'rescale' (line 29)
        rescale_114047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 36), 'rescale', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 28), tuple_114045, rescale_114047)
        
        # Processing the call keyword arguments (line 29)
        kwargs_114048 = {}
        # Getting the type of 'repr' (line 29)
        repr_114044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'repr', False)
        # Calling repr(args, kwargs) (line 29)
        repr_call_result_114049 = invoke(stypy.reporting.localization.Localization(__file__, 29, 22), repr_114044, *[tuple_114045], **kwargs_114048)
        
        # Assigning a type to the variable 'msg' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'msg', repr_call_result_114049)
        
        # Assigning a Call to a Name (line 30):
        
        # Assigning a Call to a Name (line 30):
        
        # Call to griddata(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_114051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        
        # Obtaining the type of the subscript
        slice_114052 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 30, 31), None, None, None)
        int_114053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 35), 'int')
        # Getting the type of 'x' (line 30)
        x_114054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___114055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 31), x_114054, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_114056 = invoke(stypy.reporting.localization.Localization(__file__, 30, 31), getitem___114055, (slice_114052, int_114053))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), tuple_114051, subscript_call_result_114056)
        # Adding element type (line 30)
        
        # Obtaining the type of the subscript
        slice_114057 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 30, 39), None, None, None)
        int_114058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'int')
        # Getting the type of 'x' (line 30)
        x_114059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___114060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 39), x_114059, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_114061 = invoke(stypy.reporting.localization.Localization(__file__, 30, 39), getitem___114060, (slice_114057, int_114058))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), tuple_114051, subscript_call_result_114061)
        
        # Getting the type of 'y' (line 30)
        y_114062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 48), 'y', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_114063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        
        # Obtaining the type of the subscript
        slice_114064 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 30, 52), None, None, None)
        int_114065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 56), 'int')
        # Getting the type of 'x' (line 30)
        x_114066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 52), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___114067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 52), x_114066, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_114068 = invoke(stypy.reporting.localization.Localization(__file__, 30, 52), getitem___114067, (slice_114064, int_114065))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 52), tuple_114063, subscript_call_result_114068)
        # Adding element type (line 30)
        
        # Obtaining the type of the subscript
        slice_114069 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 30, 60), None, None, None)
        int_114070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 64), 'int')
        # Getting the type of 'x' (line 30)
        x_114071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 60), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___114072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 60), x_114071, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_114073 = invoke(stypy.reporting.localization.Localization(__file__, 30, 60), getitem___114072, (slice_114069, int_114070))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 52), tuple_114063, subscript_call_result_114073)
        
        # Processing the call keyword arguments (line 30)
        # Getting the type of 'method' (line 30)
        method_114074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 76), 'method', False)
        keyword_114075 = method_114074
        # Getting the type of 'rescale' (line 31)
        rescale_114076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 38), 'rescale', False)
        keyword_114077 = rescale_114076
        kwargs_114078 = {'rescale': keyword_114077, 'method': keyword_114075}
        # Getting the type of 'griddata' (line 30)
        griddata_114050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'griddata', False)
        # Calling griddata(args, kwargs) (line 30)
        griddata_call_result_114079 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), griddata_114050, *[tuple_114051, y_114062, tuple_114063], **kwargs_114078)
        
        # Assigning a type to the variable 'yi' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'yi', griddata_call_result_114079)
        
        # Call to assert_allclose(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'y' (line 32)
        y_114081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'y', False)
        # Getting the type of 'yi' (line 32)
        yi_114082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'yi', False)
        # Processing the call keyword arguments (line 32)
        float_114083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 44), 'float')
        keyword_114084 = float_114083
        # Getting the type of 'msg' (line 32)
        msg_114085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 59), 'msg', False)
        keyword_114086 = msg_114085
        kwargs_114087 = {'err_msg': keyword_114086, 'atol': keyword_114084}
        # Getting the type of 'assert_allclose' (line 32)
        assert_allclose_114080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 32)
        assert_allclose_call_result_114088 = invoke(stypy.reporting.localization.Localization(__file__, 32, 16), assert_allclose_114080, *[y_114081, yi_114082], **kwargs_114087)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_alternative_call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_alternative_call' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_114089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114089)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_alternative_call'
        return stypy_return_type_114089


    @norecursion
    def test_multivalue_2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_multivalue_2d'
        module_type_store = module_type_store.open_function_context('test_multivalue_2d', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_multivalue_2d')
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_multivalue_2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_multivalue_2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_multivalue_2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_multivalue_2d(...)' code ##################

        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to array(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_114092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_114093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        int_114094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 23), tuple_114093, int_114094)
        # Adding element type (line 35)
        int_114095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 23), tuple_114093, int_114095)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), list_114092, tuple_114093)
        # Adding element type (line 35)
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_114096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        float_114097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 30), tuple_114096, float_114097)
        # Adding element type (line 35)
        float_114098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 30), tuple_114096, float_114098)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), list_114092, tuple_114096)
        # Adding element type (line 35)
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_114099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        float_114100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 43), tuple_114099, float_114100)
        # Adding element type (line 35)
        float_114101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 43), tuple_114099, float_114101)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), list_114092, tuple_114099)
        # Adding element type (line 35)
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_114102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        float_114103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 55), tuple_114102, float_114103)
        # Adding element type (line 35)
        float_114104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 55), tuple_114102, float_114104)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), list_114092, tuple_114102)
        # Adding element type (line 35)
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_114105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        float_114106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 67), tuple_114105, float_114106)
        # Adding element type (line 35)
        float_114107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 67), tuple_114105, float_114107)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), list_114092, tuple_114105)
        
        # Processing the call keyword arguments (line 35)
        # Getting the type of 'np' (line 36)
        np_114108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 36)
        double_114109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), np_114108, 'double')
        keyword_114110 = double_114109
        kwargs_114111 = {'dtype': keyword_114110}
        # Getting the type of 'np' (line 35)
        np_114090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 35)
        array_114091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), np_114090, 'array')
        # Calling array(args, kwargs) (line 35)
        array_call_result_114112 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), array_114091, *[list_114092], **kwargs_114111)
        
        # Assigning a type to the variable 'x' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'x', array_call_result_114112)
        
        # Assigning a BinOp to a Name (line 37):
        
        # Assigning a BinOp to a Name (line 37):
        
        # Obtaining the type of the subscript
        slice_114113 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 13), None, None, None)
        # Getting the type of 'None' (line 37)
        None_114114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 54), 'None')
        
        # Call to arange(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Obtaining the type of the subscript
        int_114117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'int')
        # Getting the type of 'x' (line 37)
        x_114118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'x', False)
        # Obtaining the member 'shape' of a type (line 37)
        shape_114119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 23), x_114118, 'shape')
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___114120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 23), shape_114119, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
        subscript_call_result_114121 = invoke(stypy.reporting.localization.Localization(__file__, 37, 23), getitem___114120, int_114117)
        
        # Processing the call keyword arguments (line 37)
        # Getting the type of 'np' (line 37)
        np_114122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 41), 'np', False)
        # Obtaining the member 'double' of a type (line 37)
        double_114123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 41), np_114122, 'double')
        keyword_114124 = double_114123
        kwargs_114125 = {'dtype': keyword_114124}
        # Getting the type of 'np' (line 37)
        np_114115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'np', False)
        # Obtaining the member 'arange' of a type (line 37)
        arange_114116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), np_114115, 'arange')
        # Calling arange(args, kwargs) (line 37)
        arange_call_result_114126 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), arange_114116, *[subscript_call_result_114121], **kwargs_114125)
        
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___114127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), arange_call_result_114126, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
        subscript_call_result_114128 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), getitem___114127, (slice_114113, None_114114))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 38)
        None_114129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'None')
        slice_114130 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 38, 15), None, None, None)
        
        # Call to array(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_114133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        int_114134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), list_114133, int_114134)
        # Adding element type (line 38)
        int_114135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), list_114133, int_114135)
        
        # Processing the call keyword arguments (line 38)
        kwargs_114136 = {}
        # Getting the type of 'np' (line 38)
        np_114131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 38)
        array_114132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), np_114131, 'array')
        # Calling array(args, kwargs) (line 38)
        array_call_result_114137 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), array_114132, *[list_114133], **kwargs_114136)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___114138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), array_call_result_114137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_114139 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), getitem___114138, (None_114129, slice_114130))
        
        # Applying the binary operator '+' (line 37)
        result_add_114140 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 13), '+', subscript_call_result_114128, subscript_call_result_114139)
        
        # Assigning a type to the variable 'y' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'y', result_add_114140)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_114141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        str_114142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'str', 'nearest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), tuple_114141, str_114142)
        # Adding element type (line 40)
        str_114143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'str', 'linear')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), tuple_114141, str_114143)
        # Adding element type (line 40)
        str_114144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 44), 'str', 'cubic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), tuple_114141, str_114144)
        
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), tuple_114141)
        # Getting the type of the for loop variable (line 40)
        for_loop_var_114145 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), tuple_114141)
        # Assigning a type to the variable 'method' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'method', for_loop_var_114145)
        # SSA begins for a for statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_114146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'True' (line 41)
        True_114147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), tuple_114146, True_114147)
        # Adding element type (line 41)
        # Getting the type of 'False' (line 41)
        False_114148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), tuple_114146, False_114148)
        
        # Testing the type of a for loop iterable (line 41)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 12), tuple_114146)
        # Getting the type of the for loop variable (line 41)
        for_loop_var_114149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 12), tuple_114146)
        # Assigning a type to the variable 'rescale' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'rescale', for_loop_var_114149)
        # SSA begins for a for statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 42):
        
        # Assigning a Call to a Name (line 42):
        
        # Call to repr(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_114151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        # Getting the type of 'method' (line 42)
        method_114152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'method', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 28), tuple_114151, method_114152)
        # Adding element type (line 42)
        # Getting the type of 'rescale' (line 42)
        rescale_114153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'rescale', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 28), tuple_114151, rescale_114153)
        
        # Processing the call keyword arguments (line 42)
        kwargs_114154 = {}
        # Getting the type of 'repr' (line 42)
        repr_114150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'repr', False)
        # Calling repr(args, kwargs) (line 42)
        repr_call_result_114155 = invoke(stypy.reporting.localization.Localization(__file__, 42, 22), repr_114150, *[tuple_114151], **kwargs_114154)
        
        # Assigning a type to the variable 'msg' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'msg', repr_call_result_114155)
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to griddata(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'x' (line 43)
        x_114157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'x', False)
        # Getting the type of 'y' (line 43)
        y_114158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'y', False)
        # Getting the type of 'x' (line 43)
        x_114159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'x', False)
        # Processing the call keyword arguments (line 43)
        # Getting the type of 'method' (line 43)
        method_114160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 46), 'method', False)
        keyword_114161 = method_114160
        # Getting the type of 'rescale' (line 43)
        rescale_114162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 62), 'rescale', False)
        keyword_114163 = rescale_114162
        kwargs_114164 = {'rescale': keyword_114163, 'method': keyword_114161}
        # Getting the type of 'griddata' (line 43)
        griddata_114156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'griddata', False)
        # Calling griddata(args, kwargs) (line 43)
        griddata_call_result_114165 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), griddata_114156, *[x_114157, y_114158, x_114159], **kwargs_114164)
        
        # Assigning a type to the variable 'yi' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'yi', griddata_call_result_114165)
        
        # Call to assert_allclose(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'y' (line 44)
        y_114167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 32), 'y', False)
        # Getting the type of 'yi' (line 44)
        yi_114168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'yi', False)
        # Processing the call keyword arguments (line 44)
        float_114169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 44), 'float')
        keyword_114170 = float_114169
        # Getting the type of 'msg' (line 44)
        msg_114171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 59), 'msg', False)
        keyword_114172 = msg_114171
        kwargs_114173 = {'err_msg': keyword_114172, 'atol': keyword_114170}
        # Getting the type of 'assert_allclose' (line 44)
        assert_allclose_114166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 44)
        assert_allclose_call_result_114174 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), assert_allclose_114166, *[y_114167, yi_114168], **kwargs_114173)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_multivalue_2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_multivalue_2d' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_114175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_multivalue_2d'
        return stypy_return_type_114175


    @norecursion
    def test_multipoint_2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_multipoint_2d'
        module_type_store = module_type_store.open_function_context('test_multipoint_2d', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_multipoint_2d')
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_multipoint_2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_multipoint_2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_multipoint_2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_multipoint_2d(...)' code ##################

        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to array(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_114178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_114179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        int_114180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 23), tuple_114179, int_114180)
        # Adding element type (line 47)
        int_114181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 23), tuple_114179, int_114181)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_114178, tuple_114179)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_114182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        float_114183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), tuple_114182, float_114183)
        # Adding element type (line 47)
        float_114184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), tuple_114182, float_114184)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_114178, tuple_114182)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_114185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        float_114186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 43), tuple_114185, float_114186)
        # Adding element type (line 47)
        float_114187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 43), tuple_114185, float_114187)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_114178, tuple_114185)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_114188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        float_114189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 55), tuple_114188, float_114189)
        # Adding element type (line 47)
        float_114190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 55), tuple_114188, float_114190)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_114178, tuple_114188)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_114191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        float_114192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 67), tuple_114191, float_114192)
        # Adding element type (line 47)
        float_114193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 67), tuple_114191, float_114193)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_114178, tuple_114191)
        
        # Processing the call keyword arguments (line 47)
        # Getting the type of 'np' (line 48)
        np_114194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 48)
        double_114195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 27), np_114194, 'double')
        keyword_114196 = double_114195
        kwargs_114197 = {'dtype': keyword_114196}
        # Getting the type of 'np' (line 47)
        np_114176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 47)
        array_114177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), np_114176, 'array')
        # Calling array(args, kwargs) (line 47)
        array_call_result_114198 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), array_114177, *[list_114178], **kwargs_114197)
        
        # Assigning a type to the variable 'x' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'x', array_call_result_114198)
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to arange(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Obtaining the type of the subscript
        int_114201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'int')
        # Getting the type of 'x' (line 49)
        x_114202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 49)
        shape_114203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 22), x_114202, 'shape')
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___114204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 22), shape_114203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_114205 = invoke(stypy.reporting.localization.Localization(__file__, 49, 22), getitem___114204, int_114201)
        
        # Processing the call keyword arguments (line 49)
        # Getting the type of 'np' (line 49)
        np_114206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 49)
        double_114207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 40), np_114206, 'double')
        keyword_114208 = double_114207
        kwargs_114209 = {'dtype': keyword_114208}
        # Getting the type of 'np' (line 49)
        np_114199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 49)
        arange_114200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), np_114199, 'arange')
        # Calling arange(args, kwargs) (line 49)
        arange_call_result_114210 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), arange_114200, *[subscript_call_result_114205], **kwargs_114209)
        
        # Assigning a type to the variable 'y' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'y', arange_call_result_114210)
        
        # Assigning a BinOp to a Name (line 51):
        
        # Assigning a BinOp to a Name (line 51):
        
        # Obtaining the type of the subscript
        slice_114211 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 51, 13), None, None, None)
        # Getting the type of 'None' (line 51)
        None_114212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'None')
        slice_114213 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 51, 13), None, None, None)
        # Getting the type of 'x' (line 51)
        x_114214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'x')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___114215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), x_114214, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_114216 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), getitem___114215, (slice_114211, None_114212, slice_114213))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 51)
        None_114217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 45), 'None')
        slice_114218 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 51, 27), None, None, None)
        # Getting the type of 'None' (line 51)
        None_114219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 52), 'None')
        
        # Call to array(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_114222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        int_114223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 36), list_114222, int_114223)
        # Adding element type (line 51)
        int_114224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 36), list_114222, int_114224)
        # Adding element type (line 51)
        int_114225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 36), list_114222, int_114225)
        
        # Processing the call keyword arguments (line 51)
        kwargs_114226 = {}
        # Getting the type of 'np' (line 51)
        np_114220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 51)
        array_114221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 27), np_114220, 'array')
        # Calling array(args, kwargs) (line 51)
        array_call_result_114227 = invoke(stypy.reporting.localization.Localization(__file__, 51, 27), array_114221, *[list_114222], **kwargs_114226)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___114228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 27), array_call_result_114227, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_114229 = invoke(stypy.reporting.localization.Localization(__file__, 51, 27), getitem___114228, (None_114217, slice_114218, None_114219))
        
        # Applying the binary operator '+' (line 51)
        result_add_114230 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 13), '+', subscript_call_result_114216, subscript_call_result_114229)
        
        # Assigning a type to the variable 'xi' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'xi', result_add_114230)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_114231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        str_114232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 23), 'str', 'nearest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 23), tuple_114231, str_114232)
        # Adding element type (line 53)
        str_114233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'str', 'linear')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 23), tuple_114231, str_114233)
        # Adding element type (line 53)
        str_114234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 44), 'str', 'cubic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 23), tuple_114231, str_114234)
        
        # Testing the type of a for loop iterable (line 53)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 8), tuple_114231)
        # Getting the type of the for loop variable (line 53)
        for_loop_var_114235 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 8), tuple_114231)
        # Assigning a type to the variable 'method' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'method', for_loop_var_114235)
        # SSA begins for a for statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_114236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'True' (line 54)
        True_114237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), tuple_114236, True_114237)
        # Adding element type (line 54)
        # Getting the type of 'False' (line 54)
        False_114238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), tuple_114236, False_114238)
        
        # Testing the type of a for loop iterable (line 54)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 12), tuple_114236)
        # Getting the type of the for loop variable (line 54)
        for_loop_var_114239 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 12), tuple_114236)
        # Assigning a type to the variable 'rescale' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'rescale', for_loop_var_114239)
        # SSA begins for a for statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to repr(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'tuple' (line 55)
        tuple_114241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 55)
        # Adding element type (line 55)
        # Getting the type of 'method' (line 55)
        method_114242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'method', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), tuple_114241, method_114242)
        # Adding element type (line 55)
        # Getting the type of 'rescale' (line 55)
        rescale_114243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'rescale', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), tuple_114241, rescale_114243)
        
        # Processing the call keyword arguments (line 55)
        kwargs_114244 = {}
        # Getting the type of 'repr' (line 55)
        repr_114240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'repr', False)
        # Calling repr(args, kwargs) (line 55)
        repr_call_result_114245 = invoke(stypy.reporting.localization.Localization(__file__, 55, 22), repr_114240, *[tuple_114241], **kwargs_114244)
        
        # Assigning a type to the variable 'msg' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'msg', repr_call_result_114245)
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to griddata(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'x' (line 56)
        x_114247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'x', False)
        # Getting the type of 'y' (line 56)
        y_114248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'y', False)
        # Getting the type of 'xi' (line 56)
        xi_114249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'xi', False)
        # Processing the call keyword arguments (line 56)
        # Getting the type of 'method' (line 56)
        method_114250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'method', False)
        keyword_114251 = method_114250
        # Getting the type of 'rescale' (line 56)
        rescale_114252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 63), 'rescale', False)
        keyword_114253 = rescale_114252
        kwargs_114254 = {'rescale': keyword_114253, 'method': keyword_114251}
        # Getting the type of 'griddata' (line 56)
        griddata_114246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'griddata', False)
        # Calling griddata(args, kwargs) (line 56)
        griddata_call_result_114255 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), griddata_114246, *[x_114247, y_114248, xi_114249], **kwargs_114254)
        
        # Assigning a type to the variable 'yi' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'yi', griddata_call_result_114255)
        
        # Call to assert_equal(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'yi' (line 58)
        yi_114257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 29), 'yi', False)
        # Obtaining the member 'shape' of a type (line 58)
        shape_114258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 29), yi_114257, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 58)
        tuple_114259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 58)
        # Adding element type (line 58)
        int_114260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), tuple_114259, int_114260)
        # Adding element type (line 58)
        int_114261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), tuple_114259, int_114261)
        
        # Processing the call keyword arguments (line 58)
        # Getting the type of 'msg' (line 58)
        msg_114262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 55), 'msg', False)
        keyword_114263 = msg_114262
        kwargs_114264 = {'err_msg': keyword_114263}
        # Getting the type of 'assert_equal' (line 58)
        assert_equal_114256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 58)
        assert_equal_call_result_114265 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), assert_equal_114256, *[shape_114258, tuple_114259], **kwargs_114264)
        
        
        # Call to assert_allclose(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'yi' (line 59)
        yi_114267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'yi', False)
        
        # Call to tile(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining the type of the subscript
        slice_114270 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 59, 44), None, None, None)
        # Getting the type of 'None' (line 59)
        None_114271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 48), 'None', False)
        # Getting the type of 'y' (line 59)
        y_114272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___114273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 44), y_114272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_114274 = invoke(stypy.reporting.localization.Localization(__file__, 59, 44), getitem___114273, (slice_114270, None_114271))
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_114275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        int_114276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 56), tuple_114275, int_114276)
        # Adding element type (line 59)
        int_114277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 56), tuple_114275, int_114277)
        
        # Processing the call keyword arguments (line 59)
        kwargs_114278 = {}
        # Getting the type of 'np' (line 59)
        np_114268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'np', False)
        # Obtaining the member 'tile' of a type (line 59)
        tile_114269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 36), np_114268, 'tile')
        # Calling tile(args, kwargs) (line 59)
        tile_call_result_114279 = invoke(stypy.reporting.localization.Localization(__file__, 59, 36), tile_114269, *[subscript_call_result_114274, tuple_114275], **kwargs_114278)
        
        # Processing the call keyword arguments (line 59)
        float_114280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 37), 'float')
        keyword_114281 = float_114280
        # Getting the type of 'msg' (line 60)
        msg_114282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 52), 'msg', False)
        keyword_114283 = msg_114282
        kwargs_114284 = {'err_msg': keyword_114283, 'atol': keyword_114281}
        # Getting the type of 'assert_allclose' (line 59)
        assert_allclose_114266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 59)
        assert_allclose_call_result_114285 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), assert_allclose_114266, *[yi_114267, tile_call_result_114279], **kwargs_114284)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_multipoint_2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_multipoint_2d' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_114286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_multipoint_2d'
        return stypy_return_type_114286


    @norecursion
    def test_complex_2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_complex_2d'
        module_type_store = module_type_store.open_function_context('test_complex_2d', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_complex_2d')
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_complex_2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_complex_2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_complex_2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_complex_2d(...)' code ##################

        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to array(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_114289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_114290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        int_114291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 23), tuple_114290, int_114291)
        # Adding element type (line 63)
        int_114292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 23), tuple_114290, int_114292)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_114289, tuple_114290)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_114293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        float_114294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 30), tuple_114293, float_114294)
        # Adding element type (line 63)
        float_114295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 30), tuple_114293, float_114295)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_114289, tuple_114293)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_114296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        float_114297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 43), tuple_114296, float_114297)
        # Adding element type (line 63)
        float_114298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 43), tuple_114296, float_114298)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_114289, tuple_114296)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_114299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        float_114300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 55), tuple_114299, float_114300)
        # Adding element type (line 63)
        float_114301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 55), tuple_114299, float_114301)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_114289, tuple_114299)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_114302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        float_114303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 67), tuple_114302, float_114303)
        # Adding element type (line 63)
        float_114304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 67), tuple_114302, float_114304)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_114289, tuple_114302)
        
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'np' (line 64)
        np_114305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 64)
        double_114306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 27), np_114305, 'double')
        keyword_114307 = double_114306
        kwargs_114308 = {'dtype': keyword_114307}
        # Getting the type of 'np' (line 63)
        np_114287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 63)
        array_114288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), np_114287, 'array')
        # Calling array(args, kwargs) (line 63)
        array_call_result_114309 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), array_114288, *[list_114289], **kwargs_114308)
        
        # Assigning a type to the variable 'x' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'x', array_call_result_114309)
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to arange(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining the type of the subscript
        int_114312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
        # Getting the type of 'x' (line 65)
        x_114313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 65)
        shape_114314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 22), x_114313, 'shape')
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___114315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 22), shape_114314, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_114316 = invoke(stypy.reporting.localization.Localization(__file__, 65, 22), getitem___114315, int_114312)
        
        # Processing the call keyword arguments (line 65)
        # Getting the type of 'np' (line 65)
        np_114317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 65)
        double_114318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 40), np_114317, 'double')
        keyword_114319 = double_114318
        kwargs_114320 = {'dtype': keyword_114319}
        # Getting the type of 'np' (line 65)
        np_114310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 65)
        arange_114311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), np_114310, 'arange')
        # Calling arange(args, kwargs) (line 65)
        arange_call_result_114321 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), arange_114311, *[subscript_call_result_114316], **kwargs_114320)
        
        # Assigning a type to the variable 'y' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'y', arange_call_result_114321)
        
        # Assigning a BinOp to a Name (line 66):
        
        # Assigning a BinOp to a Name (line 66):
        # Getting the type of 'y' (line 66)
        y_114322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'y')
        complex_114323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'complex')
        
        # Obtaining the type of the subscript
        int_114324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'int')
        slice_114325 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 66, 19), None, None, int_114324)
        # Getting the type of 'y' (line 66)
        y_114326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'y')
        # Obtaining the member '__getitem__' of a type (line 66)
        getitem___114327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 19), y_114326, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 66)
        subscript_call_result_114328 = invoke(stypy.reporting.localization.Localization(__file__, 66, 19), getitem___114327, slice_114325)
        
        # Applying the binary operator '*' (line 66)
        result_mul_114329 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 16), '*', complex_114323, subscript_call_result_114328)
        
        # Applying the binary operator '-' (line 66)
        result_sub_114330 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 12), '-', y_114322, result_mul_114329)
        
        # Assigning a type to the variable 'y' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'y', result_sub_114330)
        
        # Assigning a BinOp to a Name (line 68):
        
        # Assigning a BinOp to a Name (line 68):
        
        # Obtaining the type of the subscript
        slice_114331 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 13), None, None, None)
        # Getting the type of 'None' (line 68)
        None_114332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'None')
        slice_114333 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 13), None, None, None)
        # Getting the type of 'x' (line 68)
        x_114334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'x')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___114335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 13), x_114334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_114336 = invoke(stypy.reporting.localization.Localization(__file__, 68, 13), getitem___114335, (slice_114331, None_114332, slice_114333))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 68)
        None_114337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 45), 'None')
        slice_114338 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 27), None, None, None)
        # Getting the type of 'None' (line 68)
        None_114339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 52), 'None')
        
        # Call to array(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_114342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        int_114343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 36), list_114342, int_114343)
        # Adding element type (line 68)
        int_114344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 36), list_114342, int_114344)
        # Adding element type (line 68)
        int_114345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 36), list_114342, int_114345)
        
        # Processing the call keyword arguments (line 68)
        kwargs_114346 = {}
        # Getting the type of 'np' (line 68)
        np_114340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 68)
        array_114341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 27), np_114340, 'array')
        # Calling array(args, kwargs) (line 68)
        array_call_result_114347 = invoke(stypy.reporting.localization.Localization(__file__, 68, 27), array_114341, *[list_114342], **kwargs_114346)
        
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___114348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 27), array_call_result_114347, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_114349 = invoke(stypy.reporting.localization.Localization(__file__, 68, 27), getitem___114348, (None_114337, slice_114338, None_114339))
        
        # Applying the binary operator '+' (line 68)
        result_add_114350 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 13), '+', subscript_call_result_114336, subscript_call_result_114349)
        
        # Assigning a type to the variable 'xi' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'xi', result_add_114350)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 70)
        tuple_114351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 70)
        # Adding element type (line 70)
        str_114352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'str', 'nearest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), tuple_114351, str_114352)
        # Adding element type (line 70)
        str_114353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 34), 'str', 'linear')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), tuple_114351, str_114353)
        # Adding element type (line 70)
        str_114354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 44), 'str', 'cubic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), tuple_114351, str_114354)
        
        # Testing the type of a for loop iterable (line 70)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 8), tuple_114351)
        # Getting the type of the for loop variable (line 70)
        for_loop_var_114355 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 8), tuple_114351)
        # Assigning a type to the variable 'method' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'method', for_loop_var_114355)
        # SSA begins for a for statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_114356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        # Getting the type of 'True' (line 71)
        True_114357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 28), tuple_114356, True_114357)
        # Adding element type (line 71)
        # Getting the type of 'False' (line 71)
        False_114358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 28), tuple_114356, False_114358)
        
        # Testing the type of a for loop iterable (line 71)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 12), tuple_114356)
        # Getting the type of the for loop variable (line 71)
        for_loop_var_114359 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 12), tuple_114356)
        # Assigning a type to the variable 'rescale' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'rescale', for_loop_var_114359)
        # SSA begins for a for statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to repr(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_114361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        # Getting the type of 'method' (line 72)
        method_114362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'method', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), tuple_114361, method_114362)
        # Adding element type (line 72)
        # Getting the type of 'rescale' (line 72)
        rescale_114363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'rescale', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), tuple_114361, rescale_114363)
        
        # Processing the call keyword arguments (line 72)
        kwargs_114364 = {}
        # Getting the type of 'repr' (line 72)
        repr_114360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'repr', False)
        # Calling repr(args, kwargs) (line 72)
        repr_call_result_114365 = invoke(stypy.reporting.localization.Localization(__file__, 72, 22), repr_114360, *[tuple_114361], **kwargs_114364)
        
        # Assigning a type to the variable 'msg' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'msg', repr_call_result_114365)
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to griddata(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'x' (line 73)
        x_114367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'x', False)
        # Getting the type of 'y' (line 73)
        y_114368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 33), 'y', False)
        # Getting the type of 'xi' (line 73)
        xi_114369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 36), 'xi', False)
        # Processing the call keyword arguments (line 73)
        # Getting the type of 'method' (line 73)
        method_114370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 47), 'method', False)
        keyword_114371 = method_114370
        # Getting the type of 'rescale' (line 73)
        rescale_114372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 63), 'rescale', False)
        keyword_114373 = rescale_114372
        kwargs_114374 = {'rescale': keyword_114373, 'method': keyword_114371}
        # Getting the type of 'griddata' (line 73)
        griddata_114366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 21), 'griddata', False)
        # Calling griddata(args, kwargs) (line 73)
        griddata_call_result_114375 = invoke(stypy.reporting.localization.Localization(__file__, 73, 21), griddata_114366, *[x_114367, y_114368, xi_114369], **kwargs_114374)
        
        # Assigning a type to the variable 'yi' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'yi', griddata_call_result_114375)
        
        # Call to assert_equal(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'yi' (line 75)
        yi_114377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'yi', False)
        # Obtaining the member 'shape' of a type (line 75)
        shape_114378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 29), yi_114377, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_114379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        int_114380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 40), tuple_114379, int_114380)
        # Adding element type (line 75)
        int_114381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 40), tuple_114379, int_114381)
        
        # Processing the call keyword arguments (line 75)
        # Getting the type of 'msg' (line 75)
        msg_114382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 55), 'msg', False)
        keyword_114383 = msg_114382
        kwargs_114384 = {'err_msg': keyword_114383}
        # Getting the type of 'assert_equal' (line 75)
        assert_equal_114376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 75)
        assert_equal_call_result_114385 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), assert_equal_114376, *[shape_114378, tuple_114379], **kwargs_114384)
        
        
        # Call to assert_allclose(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'yi' (line 76)
        yi_114387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'yi', False)
        
        # Call to tile(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining the type of the subscript
        slice_114390 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 44), None, None, None)
        # Getting the type of 'None' (line 76)
        None_114391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'None', False)
        # Getting the type of 'y' (line 76)
        y_114392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___114393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 44), y_114392, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_114394 = invoke(stypy.reporting.localization.Localization(__file__, 76, 44), getitem___114393, (slice_114390, None_114391))
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_114395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        int_114396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 56), tuple_114395, int_114396)
        # Adding element type (line 76)
        int_114397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 56), tuple_114395, int_114397)
        
        # Processing the call keyword arguments (line 76)
        kwargs_114398 = {}
        # Getting the type of 'np' (line 76)
        np_114388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'np', False)
        # Obtaining the member 'tile' of a type (line 76)
        tile_114389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 36), np_114388, 'tile')
        # Calling tile(args, kwargs) (line 76)
        tile_call_result_114399 = invoke(stypy.reporting.localization.Localization(__file__, 76, 36), tile_114389, *[subscript_call_result_114394, tuple_114395], **kwargs_114398)
        
        # Processing the call keyword arguments (line 76)
        float_114400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 37), 'float')
        keyword_114401 = float_114400
        # Getting the type of 'msg' (line 77)
        msg_114402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 52), 'msg', False)
        keyword_114403 = msg_114402
        kwargs_114404 = {'err_msg': keyword_114403, 'atol': keyword_114401}
        # Getting the type of 'assert_allclose' (line 76)
        assert_allclose_114386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 76)
        assert_allclose_call_result_114405 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), assert_allclose_114386, *[yi_114387, tile_call_result_114399], **kwargs_114404)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_complex_2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_complex_2d' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_114406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_complex_2d'
        return stypy_return_type_114406


    @norecursion
    def test_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1d'
        module_type_store = module_type_store.open_function_context('test_1d', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_1d.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_1d.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_1d')
        TestGriddata.test_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1d(...)' code ##################

        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to array(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_114409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_114410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_114409, int_114410)
        # Adding element type (line 80)
        float_114411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_114409, float_114411)
        # Adding element type (line 80)
        int_114412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_114409, int_114412)
        # Adding element type (line 80)
        float_114413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_114409, float_114413)
        # Adding element type (line 80)
        int_114414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_114409, int_114414)
        # Adding element type (line 80)
        int_114415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_114409, int_114415)
        
        # Processing the call keyword arguments (line 80)
        kwargs_114416 = {}
        # Getting the type of 'np' (line 80)
        np_114407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 80)
        array_114408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), np_114407, 'array')
        # Calling array(args, kwargs) (line 80)
        array_call_result_114417 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), array_114408, *[list_114409], **kwargs_114416)
        
        # Assigning a type to the variable 'x' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'x', array_call_result_114417)
        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to array(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_114420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        int_114421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_114420, int_114421)
        # Adding element type (line 81)
        int_114422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_114420, int_114422)
        # Adding element type (line 81)
        int_114423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_114420, int_114423)
        # Adding element type (line 81)
        float_114424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_114420, float_114424)
        # Adding element type (line 81)
        int_114425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_114420, int_114425)
        # Adding element type (line 81)
        int_114426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_114420, int_114426)
        
        # Processing the call keyword arguments (line 81)
        kwargs_114427 = {}
        # Getting the type of 'np' (line 81)
        np_114418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 81)
        array_114419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), np_114418, 'array')
        # Calling array(args, kwargs) (line 81)
        array_call_result_114428 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), array_114419, *[list_114420], **kwargs_114427)
        
        # Assigning a type to the variable 'y' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'y', array_call_result_114428)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_114429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        str_114430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'str', 'nearest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 23), tuple_114429, str_114430)
        # Adding element type (line 83)
        str_114431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 34), 'str', 'linear')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 23), tuple_114429, str_114431)
        # Adding element type (line 83)
        str_114432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 44), 'str', 'cubic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 23), tuple_114429, str_114432)
        
        # Testing the type of a for loop iterable (line 83)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 83, 8), tuple_114429)
        # Getting the type of the for loop variable (line 83)
        for_loop_var_114433 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 83, 8), tuple_114429)
        # Assigning a type to the variable 'method' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'method', for_loop_var_114433)
        # SSA begins for a for statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to griddata(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'x' (line 84)
        x_114436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 37), 'x', False)
        # Getting the type of 'y' (line 84)
        y_114437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 40), 'y', False)
        # Getting the type of 'x' (line 84)
        x_114438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 43), 'x', False)
        # Processing the call keyword arguments (line 84)
        # Getting the type of 'method' (line 84)
        method_114439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 53), 'method', False)
        keyword_114440 = method_114439
        kwargs_114441 = {'method': keyword_114440}
        # Getting the type of 'griddata' (line 84)
        griddata_114435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'griddata', False)
        # Calling griddata(args, kwargs) (line 84)
        griddata_call_result_114442 = invoke(stypy.reporting.localization.Localization(__file__, 84, 28), griddata_114435, *[x_114436, y_114437, x_114438], **kwargs_114441)
        
        # Getting the type of 'y' (line 84)
        y_114443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 62), 'y', False)
        # Processing the call keyword arguments (line 84)
        # Getting the type of 'method' (line 85)
        method_114444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'method', False)
        keyword_114445 = method_114444
        float_114446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 49), 'float')
        keyword_114447 = float_114446
        kwargs_114448 = {'err_msg': keyword_114445, 'atol': keyword_114447}
        # Getting the type of 'assert_allclose' (line 84)
        assert_allclose_114434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 84)
        assert_allclose_call_result_114449 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), assert_allclose_114434, *[griddata_call_result_114442, y_114443], **kwargs_114448)
        
        
        # Call to assert_allclose(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to griddata(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to reshape(...): (line 86)
        # Processing the call arguments (line 86)
        int_114454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 47), 'int')
        int_114455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 50), 'int')
        # Processing the call keyword arguments (line 86)
        kwargs_114456 = {}
        # Getting the type of 'x' (line 86)
        x_114452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'x', False)
        # Obtaining the member 'reshape' of a type (line 86)
        reshape_114453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 37), x_114452, 'reshape')
        # Calling reshape(args, kwargs) (line 86)
        reshape_call_result_114457 = invoke(stypy.reporting.localization.Localization(__file__, 86, 37), reshape_114453, *[int_114454, int_114455], **kwargs_114456)
        
        # Getting the type of 'y' (line 86)
        y_114458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 54), 'y', False)
        # Getting the type of 'x' (line 86)
        x_114459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 57), 'x', False)
        # Processing the call keyword arguments (line 86)
        # Getting the type of 'method' (line 86)
        method_114460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 67), 'method', False)
        keyword_114461 = method_114460
        kwargs_114462 = {'method': keyword_114461}
        # Getting the type of 'griddata' (line 86)
        griddata_114451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 28), 'griddata', False)
        # Calling griddata(args, kwargs) (line 86)
        griddata_call_result_114463 = invoke(stypy.reporting.localization.Localization(__file__, 86, 28), griddata_114451, *[reshape_call_result_114457, y_114458, x_114459], **kwargs_114462)
        
        # Getting the type of 'y' (line 86)
        y_114464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 76), 'y', False)
        # Processing the call keyword arguments (line 86)
        # Getting the type of 'method' (line 87)
        method_114465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 36), 'method', False)
        keyword_114466 = method_114465
        float_114467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 49), 'float')
        keyword_114468 = float_114467
        kwargs_114469 = {'err_msg': keyword_114466, 'atol': keyword_114468}
        # Getting the type of 'assert_allclose' (line 86)
        assert_allclose_114450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 86)
        assert_allclose_call_result_114470 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), assert_allclose_114450, *[griddata_call_result_114463, y_114464], **kwargs_114469)
        
        
        # Call to assert_allclose(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to griddata(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_114473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        # Getting the type of 'x' (line 88)
        x_114474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 38), tuple_114473, x_114474)
        
        # Getting the type of 'y' (line 88)
        y_114475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 43), 'y', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_114476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        # Getting the type of 'x' (line 88)
        x_114477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 47), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 47), tuple_114476, x_114477)
        
        # Processing the call keyword arguments (line 88)
        # Getting the type of 'method' (line 88)
        method_114478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'method', False)
        keyword_114479 = method_114478
        kwargs_114480 = {'method': keyword_114479}
        # Getting the type of 'griddata' (line 88)
        griddata_114472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'griddata', False)
        # Calling griddata(args, kwargs) (line 88)
        griddata_call_result_114481 = invoke(stypy.reporting.localization.Localization(__file__, 88, 28), griddata_114472, *[tuple_114473, y_114475, tuple_114476], **kwargs_114480)
        
        # Getting the type of 'y' (line 88)
        y_114482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 68), 'y', False)
        # Processing the call keyword arguments (line 88)
        # Getting the type of 'method' (line 89)
        method_114483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'method', False)
        keyword_114484 = method_114483
        float_114485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 49), 'float')
        keyword_114486 = float_114485
        kwargs_114487 = {'err_msg': keyword_114484, 'atol': keyword_114486}
        # Getting the type of 'assert_allclose' (line 88)
        assert_allclose_114471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 88)
        assert_allclose_call_result_114488 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), assert_allclose_114471, *[griddata_call_result_114481, y_114482], **kwargs_114487)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_114489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1d'
        return stypy_return_type_114489


    @norecursion
    def test_1d_borders(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1d_borders'
        module_type_store = module_type_store.open_function_context('test_1d_borders', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_1d_borders')
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_1d_borders.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_1d_borders', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1d_borders', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1d_borders(...)' code ##################

        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to array(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_114492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        int_114493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_114492, int_114493)
        # Adding element type (line 94)
        float_114494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_114492, float_114494)
        # Adding element type (line 94)
        int_114495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_114492, int_114495)
        # Adding element type (line 94)
        float_114496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_114492, float_114496)
        # Adding element type (line 94)
        int_114497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_114492, int_114497)
        # Adding element type (line 94)
        int_114498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_114492, int_114498)
        
        # Processing the call keyword arguments (line 94)
        kwargs_114499 = {}
        # Getting the type of 'np' (line 94)
        np_114490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 94)
        array_114491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), np_114490, 'array')
        # Calling array(args, kwargs) (line 94)
        array_call_result_114500 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), array_114491, *[list_114492], **kwargs_114499)
        
        # Assigning a type to the variable 'x' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'x', array_call_result_114500)
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to array(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_114503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        int_114504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_114503, int_114504)
        # Adding element type (line 95)
        int_114505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_114503, int_114505)
        # Adding element type (line 95)
        int_114506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_114503, int_114506)
        # Adding element type (line 95)
        float_114507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_114503, float_114507)
        # Adding element type (line 95)
        int_114508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_114503, int_114508)
        # Adding element type (line 95)
        int_114509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_114503, int_114509)
        
        # Processing the call keyword arguments (line 95)
        kwargs_114510 = {}
        # Getting the type of 'np' (line 95)
        np_114501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 95)
        array_114502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), np_114501, 'array')
        # Calling array(args, kwargs) (line 95)
        array_call_result_114511 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), array_114502, *[list_114503], **kwargs_114510)
        
        # Assigning a type to the variable 'y' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'y', array_call_result_114511)
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to array(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_114514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        float_114515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 22), list_114514, float_114515)
        # Adding element type (line 96)
        float_114516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 22), list_114514, float_114516)
        
        # Processing the call keyword arguments (line 96)
        kwargs_114517 = {}
        # Getting the type of 'np' (line 96)
        np_114512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 96)
        array_114513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), np_114512, 'array')
        # Calling array(args, kwargs) (line 96)
        array_call_result_114518 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), array_114513, *[list_114514], **kwargs_114517)
        
        # Assigning a type to the variable 'xi' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'xi', array_call_result_114518)
        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to array(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_114521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        float_114522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 29), list_114521, float_114522)
        # Adding element type (line 97)
        float_114523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 29), list_114521, float_114523)
        
        # Processing the call keyword arguments (line 97)
        kwargs_114524 = {}
        # Getting the type of 'np' (line 97)
        np_114519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 97)
        array_114520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 20), np_114519, 'array')
        # Calling array(args, kwargs) (line 97)
        array_call_result_114525 = invoke(stypy.reporting.localization.Localization(__file__, 97, 20), array_114520, *[list_114521], **kwargs_114524)
        
        # Assigning a type to the variable 'yi_should' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'yi_should', array_call_result_114525)
        
        # Assigning a Str to a Name (line 99):
        
        # Assigning a Str to a Name (line 99):
        str_114526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 17), 'str', 'nearest')
        # Assigning a type to the variable 'method' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'method', str_114526)
        
        # Call to assert_allclose(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to griddata(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'x' (line 100)
        x_114529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 'x', False)
        # Getting the type of 'y' (line 100)
        y_114530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'y', False)
        # Getting the type of 'xi' (line 100)
        xi_114531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'xi', False)
        # Processing the call keyword arguments (line 100)
        # Getting the type of 'method' (line 101)
        method_114532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'method', False)
        keyword_114533 = method_114532
        kwargs_114534 = {'method': keyword_114533}
        # Getting the type of 'griddata' (line 100)
        griddata_114528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'griddata', False)
        # Calling griddata(args, kwargs) (line 100)
        griddata_call_result_114535 = invoke(stypy.reporting.localization.Localization(__file__, 100, 24), griddata_114528, *[x_114529, y_114530, xi_114531], **kwargs_114534)
        
        # Getting the type of 'yi_should' (line 101)
        yi_should_114536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 49), 'yi_should', False)
        # Processing the call keyword arguments (line 100)
        # Getting the type of 'method' (line 102)
        method_114537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 32), 'method', False)
        keyword_114538 = method_114537
        float_114539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 29), 'float')
        keyword_114540 = float_114539
        kwargs_114541 = {'err_msg': keyword_114538, 'atol': keyword_114540}
        # Getting the type of 'assert_allclose' (line 100)
        assert_allclose_114527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 100)
        assert_allclose_call_result_114542 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), assert_allclose_114527, *[griddata_call_result_114535, yi_should_114536], **kwargs_114541)
        
        
        # Call to assert_allclose(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to griddata(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to reshape(...): (line 104)
        # Processing the call arguments (line 104)
        int_114547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 43), 'int')
        int_114548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 46), 'int')
        # Processing the call keyword arguments (line 104)
        kwargs_114549 = {}
        # Getting the type of 'x' (line 104)
        x_114545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'x', False)
        # Obtaining the member 'reshape' of a type (line 104)
        reshape_114546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 33), x_114545, 'reshape')
        # Calling reshape(args, kwargs) (line 104)
        reshape_call_result_114550 = invoke(stypy.reporting.localization.Localization(__file__, 104, 33), reshape_114546, *[int_114547, int_114548], **kwargs_114549)
        
        # Getting the type of 'y' (line 104)
        y_114551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 50), 'y', False)
        # Getting the type of 'xi' (line 104)
        xi_114552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 53), 'xi', False)
        # Processing the call keyword arguments (line 104)
        # Getting the type of 'method' (line 105)
        method_114553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'method', False)
        keyword_114554 = method_114553
        kwargs_114555 = {'method': keyword_114554}
        # Getting the type of 'griddata' (line 104)
        griddata_114544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'griddata', False)
        # Calling griddata(args, kwargs) (line 104)
        griddata_call_result_114556 = invoke(stypy.reporting.localization.Localization(__file__, 104, 24), griddata_114544, *[reshape_call_result_114550, y_114551, xi_114552], **kwargs_114555)
        
        # Getting the type of 'yi_should' (line 105)
        yi_should_114557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 49), 'yi_should', False)
        # Processing the call keyword arguments (line 104)
        # Getting the type of 'method' (line 106)
        method_114558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 32), 'method', False)
        keyword_114559 = method_114558
        float_114560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 29), 'float')
        keyword_114561 = float_114560
        kwargs_114562 = {'err_msg': keyword_114559, 'atol': keyword_114561}
        # Getting the type of 'assert_allclose' (line 104)
        assert_allclose_114543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 104)
        assert_allclose_call_result_114563 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assert_allclose_114543, *[griddata_call_result_114556, yi_should_114557], **kwargs_114562)
        
        
        # Call to assert_allclose(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to griddata(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_114566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'x' (line 108)
        x_114567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 34), tuple_114566, x_114567)
        
        # Getting the type of 'y' (line 108)
        y_114568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'y', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_114569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'xi' (line 108)
        xi_114570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'xi', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 44), tuple_114569, xi_114570)
        
        # Processing the call keyword arguments (line 108)
        # Getting the type of 'method' (line 109)
        method_114571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'method', False)
        keyword_114572 = method_114571
        kwargs_114573 = {'method': keyword_114572}
        # Getting the type of 'griddata' (line 108)
        griddata_114565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'griddata', False)
        # Calling griddata(args, kwargs) (line 108)
        griddata_call_result_114574 = invoke(stypy.reporting.localization.Localization(__file__, 108, 24), griddata_114565, *[tuple_114566, y_114568, tuple_114569], **kwargs_114573)
        
        # Getting the type of 'yi_should' (line 109)
        yi_should_114575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'yi_should', False)
        # Processing the call keyword arguments (line 108)
        # Getting the type of 'method' (line 110)
        method_114576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'method', False)
        keyword_114577 = method_114576
        float_114578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'float')
        keyword_114579 = float_114578
        kwargs_114580 = {'err_msg': keyword_114577, 'atol': keyword_114579}
        # Getting the type of 'assert_allclose' (line 108)
        assert_allclose_114564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 108)
        assert_allclose_call_result_114581 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assert_allclose_114564, *[griddata_call_result_114574, yi_should_114575], **kwargs_114580)
        
        
        # ################# End of 'test_1d_borders(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1d_borders' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_114582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1d_borders'
        return stypy_return_type_114582


    @norecursion
    def test_1d_unsorted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1d_unsorted'
        module_type_store = module_type_store.open_function_context('test_1d_unsorted', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_1d_unsorted')
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_1d_unsorted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_1d_unsorted', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1d_unsorted', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1d_unsorted(...)' code ##################

        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to array(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_114585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        float_114586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 21), list_114585, float_114586)
        # Adding element type (line 114)
        int_114587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 21), list_114585, int_114587)
        # Adding element type (line 114)
        float_114588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 21), list_114585, float_114588)
        # Adding element type (line 114)
        int_114589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 21), list_114585, int_114589)
        # Adding element type (line 114)
        int_114590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 21), list_114585, int_114590)
        # Adding element type (line 114)
        int_114591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 21), list_114585, int_114591)
        
        # Processing the call keyword arguments (line 114)
        kwargs_114592 = {}
        # Getting the type of 'np' (line 114)
        np_114583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 114)
        array_114584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), np_114583, 'array')
        # Calling array(args, kwargs) (line 114)
        array_call_result_114593 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), array_114584, *[list_114585], **kwargs_114592)
        
        # Assigning a type to the variable 'x' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'x', array_call_result_114593)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to array(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_114596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        int_114597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), list_114596, int_114597)
        # Adding element type (line 115)
        int_114598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), list_114596, int_114598)
        # Adding element type (line 115)
        int_114599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), list_114596, int_114599)
        # Adding element type (line 115)
        float_114600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), list_114596, float_114600)
        # Adding element type (line 115)
        int_114601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), list_114596, int_114601)
        # Adding element type (line 115)
        int_114602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), list_114596, int_114602)
        
        # Processing the call keyword arguments (line 115)
        kwargs_114603 = {}
        # Getting the type of 'np' (line 115)
        np_114594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 115)
        array_114595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), np_114594, 'array')
        # Calling array(args, kwargs) (line 115)
        array_call_result_114604 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), array_114595, *[list_114596], **kwargs_114603)
        
        # Assigning a type to the variable 'y' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'y', array_call_result_114604)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 117)
        tuple_114605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 117)
        # Adding element type (line 117)
        str_114606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'str', 'nearest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_114605, str_114606)
        # Adding element type (line 117)
        str_114607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 34), 'str', 'linear')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_114605, str_114607)
        # Adding element type (line 117)
        str_114608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'str', 'cubic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_114605, str_114608)
        
        # Testing the type of a for loop iterable (line 117)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 8), tuple_114605)
        # Getting the type of the for loop variable (line 117)
        for_loop_var_114609 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 8), tuple_114605)
        # Assigning a type to the variable 'method' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'method', for_loop_var_114609)
        # SSA begins for a for statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to griddata(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'x' (line 118)
        x_114612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 37), 'x', False)
        # Getting the type of 'y' (line 118)
        y_114613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 40), 'y', False)
        # Getting the type of 'x' (line 118)
        x_114614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 43), 'x', False)
        # Processing the call keyword arguments (line 118)
        # Getting the type of 'method' (line 118)
        method_114615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 53), 'method', False)
        keyword_114616 = method_114615
        kwargs_114617 = {'method': keyword_114616}
        # Getting the type of 'griddata' (line 118)
        griddata_114611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'griddata', False)
        # Calling griddata(args, kwargs) (line 118)
        griddata_call_result_114618 = invoke(stypy.reporting.localization.Localization(__file__, 118, 28), griddata_114611, *[x_114612, y_114613, x_114614], **kwargs_114617)
        
        # Getting the type of 'y' (line 118)
        y_114619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 62), 'y', False)
        # Processing the call keyword arguments (line 118)
        # Getting the type of 'method' (line 119)
        method_114620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), 'method', False)
        keyword_114621 = method_114620
        float_114622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 49), 'float')
        keyword_114623 = float_114622
        kwargs_114624 = {'err_msg': keyword_114621, 'atol': keyword_114623}
        # Getting the type of 'assert_allclose' (line 118)
        assert_allclose_114610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 118)
        assert_allclose_call_result_114625 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), assert_allclose_114610, *[griddata_call_result_114618, y_114619], **kwargs_114624)
        
        
        # Call to assert_allclose(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to griddata(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to reshape(...): (line 120)
        # Processing the call arguments (line 120)
        int_114630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 47), 'int')
        int_114631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 50), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_114632 = {}
        # Getting the type of 'x' (line 120)
        x_114628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'x', False)
        # Obtaining the member 'reshape' of a type (line 120)
        reshape_114629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 37), x_114628, 'reshape')
        # Calling reshape(args, kwargs) (line 120)
        reshape_call_result_114633 = invoke(stypy.reporting.localization.Localization(__file__, 120, 37), reshape_114629, *[int_114630, int_114631], **kwargs_114632)
        
        # Getting the type of 'y' (line 120)
        y_114634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 54), 'y', False)
        # Getting the type of 'x' (line 120)
        x_114635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 57), 'x', False)
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'method' (line 120)
        method_114636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 67), 'method', False)
        keyword_114637 = method_114636
        kwargs_114638 = {'method': keyword_114637}
        # Getting the type of 'griddata' (line 120)
        griddata_114627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'griddata', False)
        # Calling griddata(args, kwargs) (line 120)
        griddata_call_result_114639 = invoke(stypy.reporting.localization.Localization(__file__, 120, 28), griddata_114627, *[reshape_call_result_114633, y_114634, x_114635], **kwargs_114638)
        
        # Getting the type of 'y' (line 120)
        y_114640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 76), 'y', False)
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'method' (line 121)
        method_114641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 36), 'method', False)
        keyword_114642 = method_114641
        float_114643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 49), 'float')
        keyword_114644 = float_114643
        kwargs_114645 = {'err_msg': keyword_114642, 'atol': keyword_114644}
        # Getting the type of 'assert_allclose' (line 120)
        assert_allclose_114626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 120)
        assert_allclose_call_result_114646 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), assert_allclose_114626, *[griddata_call_result_114639, y_114640], **kwargs_114645)
        
        
        # Call to assert_allclose(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to griddata(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining an instance of the builtin type 'tuple' (line 122)
        tuple_114649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 122)
        # Adding element type (line 122)
        # Getting the type of 'x' (line 122)
        x_114650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 38), tuple_114649, x_114650)
        
        # Getting the type of 'y' (line 122)
        y_114651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 43), 'y', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 122)
        tuple_114652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 122)
        # Adding element type (line 122)
        # Getting the type of 'x' (line 122)
        x_114653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 47), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 47), tuple_114652, x_114653)
        
        # Processing the call keyword arguments (line 122)
        # Getting the type of 'method' (line 122)
        method_114654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 59), 'method', False)
        keyword_114655 = method_114654
        kwargs_114656 = {'method': keyword_114655}
        # Getting the type of 'griddata' (line 122)
        griddata_114648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'griddata', False)
        # Calling griddata(args, kwargs) (line 122)
        griddata_call_result_114657 = invoke(stypy.reporting.localization.Localization(__file__, 122, 28), griddata_114648, *[tuple_114649, y_114651, tuple_114652], **kwargs_114656)
        
        # Getting the type of 'y' (line 122)
        y_114658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 68), 'y', False)
        # Processing the call keyword arguments (line 122)
        # Getting the type of 'method' (line 123)
        method_114659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 36), 'method', False)
        keyword_114660 = method_114659
        float_114661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'float')
        keyword_114662 = float_114661
        kwargs_114663 = {'err_msg': keyword_114660, 'atol': keyword_114662}
        # Getting the type of 'assert_allclose' (line 122)
        assert_allclose_114647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 122)
        assert_allclose_call_result_114664 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), assert_allclose_114647, *[griddata_call_result_114657, y_114658], **kwargs_114663)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_1d_unsorted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1d_unsorted' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_114665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114665)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1d_unsorted'
        return stypy_return_type_114665


    @norecursion
    def test_square_rescale_manual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_square_rescale_manual'
        module_type_store = module_type_store.open_function_context('test_square_rescale_manual', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_square_rescale_manual')
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_square_rescale_manual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_square_rescale_manual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_square_rescale_manual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_square_rescale_manual(...)' code ##################

        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to array(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_114668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_114669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        int_114670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 28), tuple_114669, int_114670)
        # Adding element type (line 126)
        int_114671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 28), tuple_114669, int_114671)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), list_114668, tuple_114669)
        # Adding element type (line 126)
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_114672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        int_114673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 35), tuple_114672, int_114673)
        # Adding element type (line 126)
        int_114674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 35), tuple_114672, int_114674)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), list_114668, tuple_114672)
        # Adding element type (line 126)
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_114675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        int_114676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 44), tuple_114675, int_114676)
        # Adding element type (line 126)
        int_114677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 44), tuple_114675, int_114677)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), list_114668, tuple_114675)
        # Adding element type (line 126)
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_114678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        int_114679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 54), tuple_114678, int_114679)
        # Adding element type (line 126)
        int_114680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 54), tuple_114678, int_114680)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), list_114668, tuple_114678)
        # Adding element type (line 126)
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_114681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        int_114682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 62), tuple_114681, int_114682)
        # Adding element type (line 126)
        int_114683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 62), tuple_114681, int_114683)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), list_114668, tuple_114681)
        
        # Processing the call keyword arguments (line 126)
        # Getting the type of 'np' (line 126)
        np_114684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 76), 'np', False)
        # Obtaining the member 'double' of a type (line 126)
        double_114685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 76), np_114684, 'double')
        keyword_114686 = double_114685
        kwargs_114687 = {'dtype': keyword_114686}
        # Getting the type of 'np' (line 126)
        np_114666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 126)
        array_114667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), np_114666, 'array')
        # Calling array(args, kwargs) (line 126)
        array_call_result_114688 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), array_114667, *[list_114668], **kwargs_114687)
        
        # Assigning a type to the variable 'points' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'points', array_call_result_114688)
        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to array(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_114691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_114692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        int_114693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 37), tuple_114692, int_114693)
        # Adding element type (line 127)
        int_114694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 37), tuple_114692, int_114694)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 35), list_114691, tuple_114692)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_114695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        int_114696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), tuple_114695, int_114696)
        # Adding element type (line 127)
        int_114697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), tuple_114695, int_114697)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 35), list_114691, tuple_114695)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_114698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        int_114699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 51), tuple_114698, int_114699)
        # Adding element type (line 127)
        int_114700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 51), tuple_114698, int_114700)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 35), list_114691, tuple_114698)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_114701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        int_114702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 58), tuple_114701, int_114702)
        # Adding element type (line 127)
        int_114703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 58), tuple_114701, int_114703)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 35), list_114691, tuple_114701)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_114704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 65), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        float_114705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 65), tuple_114704, float_114705)
        # Adding element type (line 127)
        float_114706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 65), tuple_114704, float_114706)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 35), list_114691, tuple_114704)
        
        # Processing the call keyword arguments (line 127)
        # Getting the type of 'np' (line 127)
        np_114707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 84), 'np', False)
        # Obtaining the member 'double' of a type (line 127)
        double_114708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 84), np_114707, 'double')
        keyword_114709 = double_114708
        kwargs_114710 = {'dtype': keyword_114709}
        # Getting the type of 'np' (line 127)
        np_114689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'np', False)
        # Obtaining the member 'array' of a type (line 127)
        array_114690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 26), np_114689, 'array')
        # Calling array(args, kwargs) (line 127)
        array_call_result_114711 = invoke(stypy.reporting.localization.Localization(__file__, 127, 26), array_114690, *[list_114691], **kwargs_114710)
        
        # Assigning a type to the variable 'points_rescaled' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'points_rescaled', array_call_result_114711)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to array(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_114714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        float_114715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 26), list_114714, float_114715)
        # Adding element type (line 128)
        float_114716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 26), list_114714, float_114716)
        # Adding element type (line 128)
        float_114717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 26), list_114714, float_114717)
        # Adding element type (line 128)
        float_114718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 26), list_114714, float_114718)
        # Adding element type (line 128)
        float_114719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 26), list_114714, float_114719)
        
        # Processing the call keyword arguments (line 128)
        # Getting the type of 'np' (line 128)
        np_114720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 55), 'np', False)
        # Obtaining the member 'double' of a type (line 128)
        double_114721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 55), np_114720, 'double')
        keyword_114722 = double_114721
        kwargs_114723 = {'dtype': keyword_114722}
        # Getting the type of 'np' (line 128)
        np_114712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 128)
        array_114713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 17), np_114712, 'array')
        # Calling array(args, kwargs) (line 128)
        array_call_result_114724 = invoke(stypy.reporting.localization.Localization(__file__, 128, 17), array_114713, *[list_114714], **kwargs_114723)
        
        # Assigning a type to the variable 'values' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'values', array_call_result_114724)
        
        # Assigning a Call to a Tuple (line 130):
        
        # Assigning a Subscript to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_114725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining the type of the subscript
        slice_114728 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 130, 37), None, None, None)
        # Getting the type of 'None' (line 130)
        None_114729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 62), 'None', False)
        
        # Call to linspace(...): (line 130)
        # Processing the call arguments (line 130)
        int_114732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 49), 'int')
        int_114733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 52), 'int')
        int_114734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 56), 'int')
        # Processing the call keyword arguments (line 130)
        kwargs_114735 = {}
        # Getting the type of 'np' (line 130)
        np_114730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 130)
        linspace_114731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 37), np_114730, 'linspace')
        # Calling linspace(args, kwargs) (line 130)
        linspace_call_result_114736 = invoke(stypy.reporting.localization.Localization(__file__, 130, 37), linspace_114731, *[int_114732, int_114733, int_114734], **kwargs_114735)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___114737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 37), linspace_call_result_114736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_114738 = invoke(stypy.reporting.localization.Localization(__file__, 130, 37), getitem___114737, (slice_114728, None_114729))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 131)
        None_114739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 61), 'None', False)
        slice_114740 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 131, 37), None, None, None)
        
        # Call to linspace(...): (line 131)
        # Processing the call arguments (line 131)
        int_114743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 49), 'int')
        int_114744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 52), 'int')
        int_114745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 57), 'int')
        # Processing the call keyword arguments (line 131)
        kwargs_114746 = {}
        # Getting the type of 'np' (line 131)
        np_114741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 131)
        linspace_114742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 37), np_114741, 'linspace')
        # Calling linspace(args, kwargs) (line 131)
        linspace_call_result_114747 = invoke(stypy.reporting.localization.Localization(__file__, 131, 37), linspace_114742, *[int_114743, int_114744, int_114745], **kwargs_114746)
        
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___114748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 37), linspace_call_result_114747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_114749 = invoke(stypy.reporting.localization.Localization(__file__, 131, 37), getitem___114748, (None_114739, slice_114740))
        
        # Processing the call keyword arguments (line 130)
        kwargs_114750 = {}
        # Getting the type of 'np' (line 130)
        np_114726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 130)
        broadcast_arrays_114727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 17), np_114726, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 130)
        broadcast_arrays_call_result_114751 = invoke(stypy.reporting.localization.Localization(__file__, 130, 17), broadcast_arrays_114727, *[subscript_call_result_114738, subscript_call_result_114749], **kwargs_114750)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___114752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), broadcast_arrays_call_result_114751, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_114753 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), getitem___114752, int_114725)
        
        # Assigning a type to the variable 'tuple_var_assignment_113907' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_113907', subscript_call_result_114753)
        
        # Assigning a Subscript to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_114754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining the type of the subscript
        slice_114757 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 130, 37), None, None, None)
        # Getting the type of 'None' (line 130)
        None_114758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 62), 'None', False)
        
        # Call to linspace(...): (line 130)
        # Processing the call arguments (line 130)
        int_114761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 49), 'int')
        int_114762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 52), 'int')
        int_114763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 56), 'int')
        # Processing the call keyword arguments (line 130)
        kwargs_114764 = {}
        # Getting the type of 'np' (line 130)
        np_114759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 130)
        linspace_114760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 37), np_114759, 'linspace')
        # Calling linspace(args, kwargs) (line 130)
        linspace_call_result_114765 = invoke(stypy.reporting.localization.Localization(__file__, 130, 37), linspace_114760, *[int_114761, int_114762, int_114763], **kwargs_114764)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___114766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 37), linspace_call_result_114765, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_114767 = invoke(stypy.reporting.localization.Localization(__file__, 130, 37), getitem___114766, (slice_114757, None_114758))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 131)
        None_114768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 61), 'None', False)
        slice_114769 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 131, 37), None, None, None)
        
        # Call to linspace(...): (line 131)
        # Processing the call arguments (line 131)
        int_114772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 49), 'int')
        int_114773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 52), 'int')
        int_114774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 57), 'int')
        # Processing the call keyword arguments (line 131)
        kwargs_114775 = {}
        # Getting the type of 'np' (line 131)
        np_114770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 131)
        linspace_114771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 37), np_114770, 'linspace')
        # Calling linspace(args, kwargs) (line 131)
        linspace_call_result_114776 = invoke(stypy.reporting.localization.Localization(__file__, 131, 37), linspace_114771, *[int_114772, int_114773, int_114774], **kwargs_114775)
        
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___114777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 37), linspace_call_result_114776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_114778 = invoke(stypy.reporting.localization.Localization(__file__, 131, 37), getitem___114777, (None_114768, slice_114769))
        
        # Processing the call keyword arguments (line 130)
        kwargs_114779 = {}
        # Getting the type of 'np' (line 130)
        np_114755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 130)
        broadcast_arrays_114756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 17), np_114755, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 130)
        broadcast_arrays_call_result_114780 = invoke(stypy.reporting.localization.Localization(__file__, 130, 17), broadcast_arrays_114756, *[subscript_call_result_114767, subscript_call_result_114778], **kwargs_114779)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___114781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), broadcast_arrays_call_result_114780, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_114782 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), getitem___114781, int_114754)
        
        # Assigning a type to the variable 'tuple_var_assignment_113908' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_113908', subscript_call_result_114782)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'tuple_var_assignment_113907' (line 130)
        tuple_var_assignment_113907_114783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_113907')
        # Assigning a type to the variable 'xx' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'xx', tuple_var_assignment_113907_114783)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'tuple_var_assignment_113908' (line 130)
        tuple_var_assignment_113908_114784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_113908')
        # Assigning a type to the variable 'yy' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'yy', tuple_var_assignment_113908_114784)
        
        # Assigning a Call to a Name (line 132):
        
        # Assigning a Call to a Name (line 132):
        
        # Call to ravel(...): (line 132)
        # Processing the call keyword arguments (line 132)
        kwargs_114787 = {}
        # Getting the type of 'xx' (line 132)
        xx_114785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 13), 'xx', False)
        # Obtaining the member 'ravel' of a type (line 132)
        ravel_114786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 13), xx_114785, 'ravel')
        # Calling ravel(args, kwargs) (line 132)
        ravel_call_result_114788 = invoke(stypy.reporting.localization.Localization(__file__, 132, 13), ravel_114786, *[], **kwargs_114787)
        
        # Assigning a type to the variable 'xx' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'xx', ravel_call_result_114788)
        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to ravel(...): (line 133)
        # Processing the call keyword arguments (line 133)
        kwargs_114791 = {}
        # Getting the type of 'yy' (line 133)
        yy_114789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'yy', False)
        # Obtaining the member 'ravel' of a type (line 133)
        ravel_114790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 13), yy_114789, 'ravel')
        # Calling ravel(args, kwargs) (line 133)
        ravel_call_result_114792 = invoke(stypy.reporting.localization.Localization(__file__, 133, 13), ravel_114790, *[], **kwargs_114791)
        
        # Assigning a type to the variable 'yy' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'yy', ravel_call_result_114792)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to copy(...): (line 134)
        # Processing the call keyword arguments (line 134)
        kwargs_114802 = {}
        
        # Call to array(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_114795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        # Getting the type of 'xx' (line 134)
        xx_114796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'xx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_114795, xx_114796)
        # Adding element type (line 134)
        # Getting the type of 'yy' (line 134)
        yy_114797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'yy', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_114795, yy_114797)
        
        # Processing the call keyword arguments (line 134)
        kwargs_114798 = {}
        # Getting the type of 'np' (line 134)
        np_114793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 134)
        array_114794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 13), np_114793, 'array')
        # Calling array(args, kwargs) (line 134)
        array_call_result_114799 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), array_114794, *[list_114795], **kwargs_114798)
        
        # Obtaining the member 'T' of a type (line 134)
        T_114800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 13), array_call_result_114799, 'T')
        # Obtaining the member 'copy' of a type (line 134)
        copy_114801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 13), T_114800, 'copy')
        # Calling copy(args, kwargs) (line 134)
        copy_call_result_114803 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), copy_114801, *[], **kwargs_114802)
        
        # Assigning a type to the variable 'xi' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'xi', copy_call_result_114803)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 136)
        tuple_114804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 136)
        # Adding element type (line 136)
        str_114805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'str', 'nearest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 23), tuple_114804, str_114805)
        # Adding element type (line 136)
        str_114806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 34), 'str', 'linear')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 23), tuple_114804, str_114806)
        # Adding element type (line 136)
        str_114807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 44), 'str', 'cubic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 23), tuple_114804, str_114807)
        
        # Testing the type of a for loop iterable (line 136)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 8), tuple_114804)
        # Getting the type of the for loop variable (line 136)
        for_loop_var_114808 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 8), tuple_114804)
        # Assigning a type to the variable 'method' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'method', for_loop_var_114808)
        # SSA begins for a for statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 137):
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'method' (line 137)
        method_114809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 18), 'method')
        # Assigning a type to the variable 'msg' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'msg', method_114809)
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to griddata(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'points_rescaled' (line 138)
        points_rescaled_114811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 26), 'points_rescaled', False)
        # Getting the type of 'values' (line 138)
        values_114812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 43), 'values', False)
        # Getting the type of 'xi' (line 138)
        xi_114813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 51), 'xi', False)
        
        # Call to array(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_114816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        int_114817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 63), list_114816, int_114817)
        # Adding element type (line 138)
        float_114818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 63), list_114816, float_114818)
        
        # Processing the call keyword arguments (line 138)
        kwargs_114819 = {}
        # Getting the type of 'np' (line 138)
        np_114814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 54), 'np', False)
        # Obtaining the member 'array' of a type (line 138)
        array_114815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 54), np_114814, 'array')
        # Calling array(args, kwargs) (line 138)
        array_call_result_114820 = invoke(stypy.reporting.localization.Localization(__file__, 138, 54), array_114815, *[list_114816], **kwargs_114819)
        
        # Applying the binary operator 'div' (line 138)
        result_div_114821 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 51), 'div', xi_114813, array_call_result_114820)
        
        # Processing the call keyword arguments (line 138)
        # Getting the type of 'method' (line 139)
        method_114822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'method', False)
        keyword_114823 = method_114822
        kwargs_114824 = {'method': keyword_114823}
        # Getting the type of 'griddata' (line 138)
        griddata_114810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 17), 'griddata', False)
        # Calling griddata(args, kwargs) (line 138)
        griddata_call_result_114825 = invoke(stypy.reporting.localization.Localization(__file__, 138, 17), griddata_114810, *[points_rescaled_114811, values_114812, result_div_114821], **kwargs_114824)
        
        # Assigning a type to the variable 'zi' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'zi', griddata_call_result_114825)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to griddata(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'points' (line 140)
        points_114827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 35), 'points', False)
        # Getting the type of 'values' (line 140)
        values_114828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'values', False)
        # Getting the type of 'xi' (line 140)
        xi_114829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 51), 'xi', False)
        # Processing the call keyword arguments (line 140)
        # Getting the type of 'method' (line 140)
        method_114830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 62), 'method', False)
        keyword_114831 = method_114830
        # Getting the type of 'True' (line 141)
        True_114832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 43), 'True', False)
        keyword_114833 = True_114832
        kwargs_114834 = {'rescale': keyword_114833, 'method': keyword_114831}
        # Getting the type of 'griddata' (line 140)
        griddata_114826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 26), 'griddata', False)
        # Calling griddata(args, kwargs) (line 140)
        griddata_call_result_114835 = invoke(stypy.reporting.localization.Localization(__file__, 140, 26), griddata_114826, *[points_114827, values_114828, xi_114829], **kwargs_114834)
        
        # Assigning a type to the variable 'zi_rescaled' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'zi_rescaled', griddata_call_result_114835)
        
        # Call to assert_allclose(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'zi' (line 142)
        zi_114837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 28), 'zi', False)
        # Getting the type of 'zi_rescaled' (line 142)
        zi_rescaled_114838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 32), 'zi_rescaled', False)
        # Processing the call keyword arguments (line 142)
        # Getting the type of 'msg' (line 142)
        msg_114839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 53), 'msg', False)
        keyword_114840 = msg_114839
        float_114841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 33), 'float')
        keyword_114842 = float_114841
        kwargs_114843 = {'err_msg': keyword_114840, 'atol': keyword_114842}
        # Getting the type of 'assert_allclose' (line 142)
        assert_allclose_114836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 142)
        assert_allclose_call_result_114844 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), assert_allclose_114836, *[zi_114837, zi_rescaled_114838], **kwargs_114843)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_square_rescale_manual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_square_rescale_manual' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_114845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114845)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_square_rescale_manual'
        return stypy_return_type_114845


    @norecursion
    def test_xi_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_xi_1d'
        module_type_store = module_type_store.open_function_context('test_xi_1d', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_localization', localization)
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_function_name', 'TestGriddata.test_xi_1d')
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGriddata.test_xi_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.test_xi_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_xi_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_xi_1d(...)' code ##################

        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to array(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_114848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_114849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        int_114850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 23), tuple_114849, int_114850)
        # Adding element type (line 147)
        int_114851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 23), tuple_114849, int_114851)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_114848, tuple_114849)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_114852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        float_114853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 30), tuple_114852, float_114853)
        # Adding element type (line 147)
        float_114854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 30), tuple_114852, float_114854)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_114848, tuple_114852)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_114855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        float_114856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 43), tuple_114855, float_114856)
        # Adding element type (line 147)
        float_114857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 43), tuple_114855, float_114857)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_114848, tuple_114855)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_114858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        float_114859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 55), tuple_114858, float_114859)
        # Adding element type (line 147)
        float_114860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 55), tuple_114858, float_114860)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_114848, tuple_114858)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_114861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        float_114862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 67), tuple_114861, float_114862)
        # Adding element type (line 147)
        float_114863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 67), tuple_114861, float_114863)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_114848, tuple_114861)
        
        # Processing the call keyword arguments (line 147)
        # Getting the type of 'np' (line 148)
        np_114864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 148)
        double_114865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 27), np_114864, 'double')
        keyword_114866 = double_114865
        kwargs_114867 = {'dtype': keyword_114866}
        # Getting the type of 'np' (line 147)
        np_114846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 147)
        array_114847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), np_114846, 'array')
        # Calling array(args, kwargs) (line 147)
        array_call_result_114868 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), array_114847, *[list_114848], **kwargs_114867)
        
        # Assigning a type to the variable 'x' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'x', array_call_result_114868)
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to arange(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining the type of the subscript
        int_114871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 30), 'int')
        # Getting the type of 'x' (line 149)
        x_114872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 149)
        shape_114873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 22), x_114872, 'shape')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___114874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 22), shape_114873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_114875 = invoke(stypy.reporting.localization.Localization(__file__, 149, 22), getitem___114874, int_114871)
        
        # Processing the call keyword arguments (line 149)
        # Getting the type of 'np' (line 149)
        np_114876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 149)
        double_114877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 40), np_114876, 'double')
        keyword_114878 = double_114877
        kwargs_114879 = {'dtype': keyword_114878}
        # Getting the type of 'np' (line 149)
        np_114869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 149)
        arange_114870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), np_114869, 'arange')
        # Calling arange(args, kwargs) (line 149)
        arange_call_result_114880 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), arange_114870, *[subscript_call_result_114875], **kwargs_114879)
        
        # Assigning a type to the variable 'y' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'y', arange_call_result_114880)
        
        # Assigning a BinOp to a Name (line 150):
        
        # Assigning a BinOp to a Name (line 150):
        # Getting the type of 'y' (line 150)
        y_114881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'y')
        complex_114882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'complex')
        
        # Obtaining the type of the subscript
        int_114883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 23), 'int')
        slice_114884 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 150, 19), None, None, int_114883)
        # Getting the type of 'y' (line 150)
        y_114885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'y')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___114886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 19), y_114885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_114887 = invoke(stypy.reporting.localization.Localization(__file__, 150, 19), getitem___114886, slice_114884)
        
        # Applying the binary operator '*' (line 150)
        result_mul_114888 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 16), '*', complex_114882, subscript_call_result_114887)
        
        # Applying the binary operator '-' (line 150)
        result_sub_114889 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '-', y_114881, result_mul_114888)
        
        # Assigning a type to the variable 'y' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'y', result_sub_114889)
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to array(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_114892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        float_114893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 22), list_114892, float_114893)
        # Adding element type (line 152)
        float_114894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 22), list_114892, float_114894)
        
        # Processing the call keyword arguments (line 152)
        kwargs_114895 = {}
        # Getting the type of 'np' (line 152)
        np_114890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 152)
        array_114891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), np_114890, 'array')
        # Calling array(args, kwargs) (line 152)
        array_call_result_114896 = invoke(stypy.reporting.localization.Localization(__file__, 152, 13), array_114891, *[list_114892], **kwargs_114895)
        
        # Assigning a type to the variable 'xi' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'xi', array_call_result_114896)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_114897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        str_114898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'str', 'nearest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 23), tuple_114897, str_114898)
        # Adding element type (line 154)
        str_114899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 34), 'str', 'linear')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 23), tuple_114897, str_114899)
        # Adding element type (line 154)
        str_114900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 44), 'str', 'cubic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 23), tuple_114897, str_114900)
        
        # Testing the type of a for loop iterable (line 154)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 154, 8), tuple_114897)
        # Getting the type of the for loop variable (line 154)
        for_loop_var_114901 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 154, 8), tuple_114897)
        # Assigning a type to the variable 'method' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'method', for_loop_var_114901)
        # SSA begins for a for statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to griddata(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'x' (line 155)
        x_114903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'x', False)
        # Getting the type of 'y' (line 155)
        y_114904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'y', False)
        # Getting the type of 'xi' (line 155)
        xi_114905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'xi', False)
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'method' (line 155)
        method_114906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 43), 'method', False)
        keyword_114907 = method_114906
        kwargs_114908 = {'method': keyword_114907}
        # Getting the type of 'griddata' (line 155)
        griddata_114902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'griddata', False)
        # Calling griddata(args, kwargs) (line 155)
        griddata_call_result_114909 = invoke(stypy.reporting.localization.Localization(__file__, 155, 17), griddata_114902, *[x_114903, y_114904, xi_114905], **kwargs_114908)
        
        # Assigning a type to the variable 'p1' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'p1', griddata_call_result_114909)
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to griddata(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'x' (line 156)
        x_114911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'x', False)
        # Getting the type of 'y' (line 156)
        y_114912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'y', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 156)
        None_114913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'None', False)
        slice_114914 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 156, 32), None, None, None)
        # Getting the type of 'xi' (line 156)
        xi_114915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'xi', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___114916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 32), xi_114915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_114917 = invoke(stypy.reporting.localization.Localization(__file__, 156, 32), getitem___114916, (None_114913, slice_114914))
        
        # Processing the call keyword arguments (line 156)
        # Getting the type of 'method' (line 156)
        method_114918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 51), 'method', False)
        keyword_114919 = method_114918
        kwargs_114920 = {'method': keyword_114919}
        # Getting the type of 'griddata' (line 156)
        griddata_114910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 17), 'griddata', False)
        # Calling griddata(args, kwargs) (line 156)
        griddata_call_result_114921 = invoke(stypy.reporting.localization.Localization(__file__, 156, 17), griddata_114910, *[x_114911, y_114912, subscript_call_result_114917], **kwargs_114920)
        
        # Assigning a type to the variable 'p2' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'p2', griddata_call_result_114921)
        
        # Call to assert_allclose(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'p1' (line 157)
        p1_114923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'p1', False)
        # Getting the type of 'p2' (line 157)
        p2_114924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'p2', False)
        # Processing the call keyword arguments (line 157)
        # Getting the type of 'method' (line 157)
        method_114925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 44), 'method', False)
        keyword_114926 = method_114925
        kwargs_114927 = {'err_msg': keyword_114926}
        # Getting the type of 'assert_allclose' (line 157)
        assert_allclose_114922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 157)
        assert_allclose_call_result_114928 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), assert_allclose_114922, *[p1_114923, p2_114924], **kwargs_114927)
        
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to array(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_114931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        float_114932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 27), list_114931, float_114932)
        
        # Processing the call keyword arguments (line 159)
        kwargs_114933 = {}
        # Getting the type of 'np' (line 159)
        np_114929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 159)
        array_114930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 18), np_114929, 'array')
        # Calling array(args, kwargs) (line 159)
        array_call_result_114934 = invoke(stypy.reporting.localization.Localization(__file__, 159, 18), array_114930, *[list_114931], **kwargs_114933)
        
        # Assigning a type to the variable 'xi1' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'xi1', array_call_result_114934)
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to array(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_114937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        float_114938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 27), list_114937, float_114938)
        # Adding element type (line 160)
        float_114939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 27), list_114937, float_114939)
        # Adding element type (line 160)
        float_114940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 27), list_114937, float_114940)
        
        # Processing the call keyword arguments (line 160)
        kwargs_114941 = {}
        # Getting the type of 'np' (line 160)
        np_114935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 160)
        array_114936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 18), np_114935, 'array')
        # Calling array(args, kwargs) (line 160)
        array_call_result_114942 = invoke(stypy.reporting.localization.Localization(__file__, 160, 18), array_114936, *[list_114937], **kwargs_114941)
        
        # Assigning a type to the variable 'xi3' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'xi3', array_call_result_114942)
        
        # Call to assert_raises(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'ValueError' (line 161)
        ValueError_114944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'ValueError', False)
        # Getting the type of 'griddata' (line 161)
        griddata_114945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 38), 'griddata', False)
        # Getting the type of 'x' (line 161)
        x_114946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 48), 'x', False)
        # Getting the type of 'y' (line 161)
        y_114947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 51), 'y', False)
        # Getting the type of 'xi1' (line 161)
        xi1_114948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 54), 'xi1', False)
        # Processing the call keyword arguments (line 161)
        # Getting the type of 'method' (line 162)
        method_114949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 33), 'method', False)
        keyword_114950 = method_114949
        kwargs_114951 = {'method': keyword_114950}
        # Getting the type of 'assert_raises' (line 161)
        assert_raises_114943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 161)
        assert_raises_call_result_114952 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), assert_raises_114943, *[ValueError_114944, griddata_114945, x_114946, y_114947, xi1_114948], **kwargs_114951)
        
        
        # Call to assert_raises(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'ValueError' (line 163)
        ValueError_114954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'ValueError', False)
        # Getting the type of 'griddata' (line 163)
        griddata_114955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 38), 'griddata', False)
        # Getting the type of 'x' (line 163)
        x_114956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 48), 'x', False)
        # Getting the type of 'y' (line 163)
        y_114957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'y', False)
        # Getting the type of 'xi3' (line 163)
        xi3_114958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 54), 'xi3', False)
        # Processing the call keyword arguments (line 163)
        # Getting the type of 'method' (line 164)
        method_114959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 33), 'method', False)
        keyword_114960 = method_114959
        kwargs_114961 = {'method': keyword_114960}
        # Getting the type of 'assert_raises' (line 163)
        assert_raises_114953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 163)
        assert_raises_call_result_114962 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), assert_raises_114953, *[ValueError_114954, griddata_114955, x_114956, y_114957, xi3_114958], **kwargs_114961)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_xi_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_xi_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_114963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_xi_1d'
        return stypy_return_type_114963


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGriddata.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestGriddata' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'TestGriddata', TestGriddata)

@norecursion
def test_nearest_options(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_nearest_options'
    module_type_store = module_type_store.open_function_context('test_nearest_options', 167, 0, False)
    
    # Passed parameters checking function
    test_nearest_options.stypy_localization = localization
    test_nearest_options.stypy_type_of_self = None
    test_nearest_options.stypy_type_store = module_type_store
    test_nearest_options.stypy_function_name = 'test_nearest_options'
    test_nearest_options.stypy_param_names_list = []
    test_nearest_options.stypy_varargs_param_name = None
    test_nearest_options.stypy_kwargs_param_name = None
    test_nearest_options.stypy_call_defaults = defaults
    test_nearest_options.stypy_call_varargs = varargs
    test_nearest_options.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_nearest_options', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_nearest_options', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_nearest_options(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 169):
    
    # Assigning a Num to a Name (line 169):
    int_114964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_113909' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_113909', int_114964)
    
    # Assigning a Num to a Name (line 169):
    int_114965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 18), 'int')
    # Assigning a type to the variable 'tuple_assignment_113910' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_113910', int_114965)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_assignment_113909' (line 169)
    tuple_assignment_113909_114966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_113909')
    # Assigning a type to the variable 'npts' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'npts', tuple_assignment_113909_114966)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_assignment_113910' (line 169)
    tuple_assignment_113910_114967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_113910')
    # Assigning a type to the variable 'nd' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 10), 'nd', tuple_assignment_113910_114967)
    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to reshape(...): (line 170)
    # Processing the call arguments (line 170)
    
    # Obtaining an instance of the builtin type 'tuple' (line 170)
    tuple_114976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 170)
    # Adding element type (line 170)
    # Getting the type of 'npts' (line 170)
    npts_114977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 36), 'npts', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 36), tuple_114976, npts_114977)
    # Adding element type (line 170)
    # Getting the type of 'nd' (line 170)
    nd_114978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 42), 'nd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 36), tuple_114976, nd_114978)
    
    # Processing the call keyword arguments (line 170)
    kwargs_114979 = {}
    
    # Call to arange(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'npts' (line 170)
    npts_114970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'npts', False)
    # Getting the type of 'nd' (line 170)
    nd_114971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'nd', False)
    # Applying the binary operator '*' (line 170)
    result_mul_114972 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 18), '*', npts_114970, nd_114971)
    
    # Processing the call keyword arguments (line 170)
    kwargs_114973 = {}
    # Getting the type of 'np' (line 170)
    np_114968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 170)
    arange_114969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), np_114968, 'arange')
    # Calling arange(args, kwargs) (line 170)
    arange_call_result_114974 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), arange_114969, *[result_mul_114972], **kwargs_114973)
    
    # Obtaining the member 'reshape' of a type (line 170)
    reshape_114975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), arange_call_result_114974, 'reshape')
    # Calling reshape(args, kwargs) (line 170)
    reshape_call_result_114980 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), reshape_114975, *[tuple_114976], **kwargs_114979)
    
    # Assigning a type to the variable 'x' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'x', reshape_call_result_114980)
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to arange(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'npts' (line 171)
    npts_114983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'npts', False)
    # Processing the call keyword arguments (line 171)
    kwargs_114984 = {}
    # Getting the type of 'np' (line 171)
    np_114981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 171)
    arange_114982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), np_114981, 'arange')
    # Calling arange(args, kwargs) (line 171)
    arange_call_result_114985 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), arange_114982, *[npts_114983], **kwargs_114984)
    
    # Assigning a type to the variable 'y' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'y', arange_call_result_114985)
    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to NearestNDInterpolator(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'x' (line 172)
    x_114987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 33), 'x', False)
    # Getting the type of 'y' (line 172)
    y_114988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 36), 'y', False)
    # Processing the call keyword arguments (line 172)
    kwargs_114989 = {}
    # Getting the type of 'NearestNDInterpolator' (line 172)
    NearestNDInterpolator_114986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'NearestNDInterpolator', False)
    # Calling NearestNDInterpolator(args, kwargs) (line 172)
    NearestNDInterpolator_call_result_114990 = invoke(stypy.reporting.localization.Localization(__file__, 172, 11), NearestNDInterpolator_114986, *[x_114987, y_114988], **kwargs_114989)
    
    # Assigning a type to the variable 'nndi' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'nndi', NearestNDInterpolator_call_result_114990)
    
    # Assigning a Dict to a Name (line 174):
    
    # Assigning a Dict to a Name (line 174):
    
    # Obtaining an instance of the builtin type 'dict' (line 174)
    dict_114991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 174)
    # Adding element type (key, value) (line 174)
    str_114992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 12), 'str', 'balanced_tree')
    # Getting the type of 'False' (line 174)
    False_114993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'False')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 11), dict_114991, (str_114992, False_114993))
    # Adding element type (key, value) (line 174)
    str_114994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 36), 'str', 'compact_nodes')
    # Getting the type of 'False' (line 174)
    False_114995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 53), 'False')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 11), dict_114991, (str_114994, False_114995))
    
    # Assigning a type to the variable 'opts' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'opts', dict_114991)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to NearestNDInterpolator(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'x' (line 175)
    x_114997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'x', False)
    # Getting the type of 'y' (line 175)
    y_114998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 38), 'y', False)
    # Processing the call keyword arguments (line 175)
    # Getting the type of 'opts' (line 175)
    opts_114999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 54), 'opts', False)
    keyword_115000 = opts_114999
    kwargs_115001 = {'tree_options': keyword_115000}
    # Getting the type of 'NearestNDInterpolator' (line 175)
    NearestNDInterpolator_114996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), 'NearestNDInterpolator', False)
    # Calling NearestNDInterpolator(args, kwargs) (line 175)
    NearestNDInterpolator_call_result_115002 = invoke(stypy.reporting.localization.Localization(__file__, 175, 13), NearestNDInterpolator_114996, *[x_114997, y_114998], **kwargs_115001)
    
    # Assigning a type to the variable 'nndi_o' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'nndi_o', NearestNDInterpolator_call_result_115002)
    
    # Call to assert_allclose(...): (line 176)
    # Processing the call arguments (line 176)
    
    # Call to nndi(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'x' (line 176)
    x_115005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'x', False)
    # Processing the call keyword arguments (line 176)
    kwargs_115006 = {}
    # Getting the type of 'nndi' (line 176)
    nndi_115004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'nndi', False)
    # Calling nndi(args, kwargs) (line 176)
    nndi_call_result_115007 = invoke(stypy.reporting.localization.Localization(__file__, 176, 20), nndi_115004, *[x_115005], **kwargs_115006)
    
    
    # Call to nndi_o(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'x' (line 176)
    x_115009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'x', False)
    # Processing the call keyword arguments (line 176)
    kwargs_115010 = {}
    # Getting the type of 'nndi_o' (line 176)
    nndi_o_115008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 29), 'nndi_o', False)
    # Calling nndi_o(args, kwargs) (line 176)
    nndi_o_call_result_115011 = invoke(stypy.reporting.localization.Localization(__file__, 176, 29), nndi_o_115008, *[x_115009], **kwargs_115010)
    
    # Processing the call keyword arguments (line 176)
    float_115012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 45), 'float')
    keyword_115013 = float_115012
    kwargs_115014 = {'atol': keyword_115013}
    # Getting the type of 'assert_allclose' (line 176)
    assert_allclose_115003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 176)
    assert_allclose_call_result_115015 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), assert_allclose_115003, *[nndi_call_result_115007, nndi_o_call_result_115011], **kwargs_115014)
    
    
    # ################# End of 'test_nearest_options(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_nearest_options' in the type store
    # Getting the type of 'stypy_return_type' (line 167)
    stypy_return_type_115016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115016)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_nearest_options'
    return stypy_return_type_115016

# Assigning a type to the variable 'test_nearest_options' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'test_nearest_options', test_nearest_options)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
