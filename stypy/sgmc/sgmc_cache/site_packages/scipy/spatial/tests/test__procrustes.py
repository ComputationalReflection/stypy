
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import absolute_import, division, print_function
2: 
3: import numpy as np
4: from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
5: from pytest import raises as assert_raises
6: 
7: from scipy.spatial import procrustes
8: 
9: 
10: class TestProcrustes(object):
11:     def setup_method(self):
12:         '''creates inputs'''
13:         # an L
14:         self.data1 = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
15: 
16:         # a larger, shifted, mirrored L
17:         self.data2 = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
18: 
19:         # an L shifted up 1, right 1, and with point 4 shifted an extra .5
20:         # to the right
21:         # pointwise distance disparity with data1: 3*(2) + (1 + 1.5^2)
22:         self.data3 = np.array([[2, 4], [2, 3], [2, 2], [3, 2.5]], 'd')
23: 
24:         # data4, data5 are standardized (trace(A*A') = 1).
25:         # procrustes should return an identical copy if they are used
26:         # as the first matrix argument.
27:         shiftangle = np.pi / 8
28:         self.data4 = np.array([[1, 0], [0, 1], [-1, 0],
29:                               [0, -1]], 'd') / np.sqrt(4)
30:         self.data5 = np.array([[np.cos(shiftangle), np.sin(shiftangle)],
31:                               [np.cos(np.pi / 2 - shiftangle),
32:                                np.sin(np.pi / 2 - shiftangle)],
33:                               [-np.cos(shiftangle),
34:                                -np.sin(shiftangle)],
35:                               [-np.cos(np.pi / 2 - shiftangle),
36:                                -np.sin(np.pi / 2 - shiftangle)]],
37:                               'd') / np.sqrt(4)
38: 
39:     def test_procrustes(self):
40:         # tests procrustes' ability to match two matrices.
41:         #
42:         # the second matrix is a rotated, shifted, scaled, and mirrored version
43:         # of the first, in two dimensions only
44:         #
45:         # can shift, mirror, and scale an 'L'?
46:         a, b, disparity = procrustes(self.data1, self.data2)
47:         assert_allclose(b, a)
48:         assert_almost_equal(disparity, 0.)
49: 
50:         # if first mtx is standardized, leaves first mtx unchanged?
51:         m4, m5, disp45 = procrustes(self.data4, self.data5)
52:         assert_equal(m4, self.data4)
53: 
54:         # at worst, data3 is an 'L' with one point off by .5
55:         m1, m3, disp13 = procrustes(self.data1, self.data3)
56:         #assert_(disp13 < 0.5 ** 2)
57: 
58:     def test_procrustes2(self):
59:         # procrustes disparity should not depend on order of matrices
60:         m1, m3, disp13 = procrustes(self.data1, self.data3)
61:         m3_2, m1_2, disp31 = procrustes(self.data3, self.data1)
62:         assert_almost_equal(disp13, disp31)
63: 
64:         # try with 3d, 8 pts per
65:         rand1 = np.array([[2.61955202, 0.30522265, 0.55515826],
66:                          [0.41124708, -0.03966978, -0.31854548],
67:                          [0.91910318, 1.39451809, -0.15295084],
68:                          [2.00452023, 0.50150048, 0.29485268],
69:                          [0.09453595, 0.67528885, 0.03283872],
70:                          [0.07015232, 2.18892599, -1.67266852],
71:                          [0.65029688, 1.60551637, 0.80013549],
72:                          [-0.6607528, 0.53644208, 0.17033891]])
73: 
74:         rand3 = np.array([[0.0809969, 0.09731461, -0.173442],
75:                          [-1.84888465, -0.92589646, -1.29335743],
76:                          [0.67031855, -1.35957463, 0.41938621],
77:                          [0.73967209, -0.20230757, 0.52418027],
78:                          [0.17752796, 0.09065607, 0.29827466],
79:                          [0.47999368, -0.88455717, -0.57547934],
80:                          [-0.11486344, -0.12608506, -0.3395779],
81:                          [-0.86106154, -0.28687488, 0.9644429]])
82:         res1, res3, disp13 = procrustes(rand1, rand3)
83:         res3_2, res1_2, disp31 = procrustes(rand3, rand1)
84:         assert_almost_equal(disp13, disp31)
85: 
86:     def test_procrustes_shape_mismatch(self):
87:         assert_raises(ValueError, procrustes,
88:                       np.array([[1, 2], [3, 4]]),
89:                       np.array([[5, 6, 7], [8, 9, 10]]))
90: 
91:     def test_procrustes_empty_rows_or_cols(self):
92:         empty = np.array([[]])
93:         assert_raises(ValueError, procrustes, empty, empty)
94: 
95:     def test_procrustes_no_variation(self):
96:         assert_raises(ValueError, procrustes,
97:                       np.array([[42, 42], [42, 42]]),
98:                       np.array([[45, 45], [45, 45]]))
99: 
100:     def test_procrustes_bad_number_of_dimensions(self):
101:         # fewer dimensions in one dataset
102:         assert_raises(ValueError, procrustes,
103:                       np.array([1, 1, 2, 3, 5, 8]),
104:                       np.array([[1, 2], [3, 4]]))
105: 
106:         # fewer dimensions in both datasets
107:         assert_raises(ValueError, procrustes,
108:                       np.array([1, 1, 2, 3, 5, 8]),
109:                       np.array([1, 1, 2, 3, 5, 8]))
110: 
111:         # zero dimensions
112:         assert_raises(ValueError, procrustes, np.array(7), np.array(11))
113: 
114:         # extra dimensions
115:         assert_raises(ValueError, procrustes,
116:                       np.array([[[11], [7]]]),
117:                       np.array([[[5, 13]]]))
118: 
119: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492590 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_492590) is not StypyTypeError):

    if (import_492590 != 'pyd_module'):
        __import__(import_492590)
        sys_modules_492591 = sys.modules[import_492590]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_492591.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_492590)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_allclose, assert_equal, assert_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492592 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_492592) is not StypyTypeError):

    if (import_492592 != 'pyd_module'):
        __import__(import_492592)
        sys_modules_492593 = sys.modules[import_492592]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_492593.module_type_store, module_type_store, ['assert_allclose', 'assert_equal', 'assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_492593, sys_modules_492593.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal, assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal', 'assert_almost_equal'], [assert_allclose, assert_equal, assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_492592)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from pytest import assert_raises' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492594 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_492594) is not StypyTypeError):

    if (import_492594 != 'pyd_module'):
        __import__(import_492594)
        sys_modules_492595 = sys.modules[import_492594]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_492595.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_492595, sys_modules_492595.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_492594)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.spatial import procrustes' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492596 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.spatial')

if (type(import_492596) is not StypyTypeError):

    if (import_492596 != 'pyd_module'):
        __import__(import_492596)
        sys_modules_492597 = sys.modules[import_492596]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.spatial', sys_modules_492597.module_type_store, module_type_store, ['procrustes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_492597, sys_modules_492597.module_type_store, module_type_store)
    else:
        from scipy.spatial import procrustes

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.spatial', None, module_type_store, ['procrustes'], [procrustes])

else:
    # Assigning a type to the variable 'scipy.spatial' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.spatial', import_492596)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

# Declaration of the 'TestProcrustes' class

class TestProcrustes(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_function_name', 'TestProcrustes.setup_method')
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProcrustes.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProcrustes.setup_method', [], None, None, defaults, varargs, kwargs)

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

        str_492598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'str', 'creates inputs')
        
        # Assigning a Call to a Attribute (line 14):
        
        # Assigning a Call to a Attribute (line 14):
        
        # Call to array(...): (line 14)
        # Processing the call arguments (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_492601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_492602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        int_492603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 31), list_492602, int_492603)
        # Adding element type (line 14)
        int_492604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 31), list_492602, int_492604)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 30), list_492601, list_492602)
        # Adding element type (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_492605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        int_492606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 39), list_492605, int_492606)
        # Adding element type (line 14)
        int_492607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 39), list_492605, int_492607)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 30), list_492601, list_492605)
        # Adding element type (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_492608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        int_492609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 47), list_492608, int_492609)
        # Adding element type (line 14)
        int_492610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 47), list_492608, int_492610)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 30), list_492601, list_492608)
        # Adding element type (line 14)
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_492611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        int_492612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 55), list_492611, int_492612)
        # Adding element type (line 14)
        int_492613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 55), list_492611, int_492613)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 30), list_492601, list_492611)
        
        str_492614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 64), 'str', 'd')
        # Processing the call keyword arguments (line 14)
        kwargs_492615 = {}
        # Getting the type of 'np' (line 14)
        np_492599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 14)
        array_492600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), np_492599, 'array')
        # Calling array(args, kwargs) (line 14)
        array_call_result_492616 = invoke(stypy.reporting.localization.Localization(__file__, 14, 21), array_492600, *[list_492601, str_492614], **kwargs_492615)
        
        # Getting the type of 'self' (line 14)
        self_492617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'data1' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_492617, 'data1', array_call_result_492616)
        
        # Assigning a Call to a Attribute (line 17):
        
        # Assigning a Call to a Attribute (line 17):
        
        # Call to array(...): (line 17)
        # Processing the call arguments (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_492620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_492621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_492622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 31), list_492621, int_492622)
        # Adding element type (line 17)
        int_492623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 31), list_492621, int_492623)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 30), list_492620, list_492621)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_492624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_492625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 40), list_492624, int_492625)
        # Adding element type (line 17)
        int_492626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 40), list_492624, int_492626)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 30), list_492620, list_492624)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_492627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_492628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 49), list_492627, int_492628)
        # Adding element type (line 17)
        int_492629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 49), list_492627, int_492629)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 30), list_492620, list_492627)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_492630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_492631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 58), list_492630, int_492631)
        # Adding element type (line 17)
        int_492632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 58), list_492630, int_492632)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 30), list_492620, list_492630)
        
        str_492633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 68), 'str', 'd')
        # Processing the call keyword arguments (line 17)
        kwargs_492634 = {}
        # Getting the type of 'np' (line 17)
        np_492618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 17)
        array_492619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 21), np_492618, 'array')
        # Calling array(args, kwargs) (line 17)
        array_call_result_492635 = invoke(stypy.reporting.localization.Localization(__file__, 17, 21), array_492619, *[list_492620, str_492633], **kwargs_492634)
        
        # Getting the type of 'self' (line 17)
        self_492636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'data2' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_492636, 'data2', array_call_result_492635)
        
        # Assigning a Call to a Attribute (line 22):
        
        # Assigning a Call to a Attribute (line 22):
        
        # Call to array(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_492639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_492640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        int_492641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), list_492640, int_492641)
        # Adding element type (line 22)
        int_492642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), list_492640, int_492642)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 30), list_492639, list_492640)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_492643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        int_492644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 39), list_492643, int_492644)
        # Adding element type (line 22)
        int_492645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 39), list_492643, int_492645)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 30), list_492639, list_492643)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_492646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        int_492647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_492646, int_492647)
        # Adding element type (line 22)
        int_492648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_492646, int_492648)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 30), list_492639, list_492646)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_492649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        int_492650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 55), list_492649, int_492650)
        # Adding element type (line 22)
        float_492651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 55), list_492649, float_492651)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 30), list_492639, list_492649)
        
        str_492652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 66), 'str', 'd')
        # Processing the call keyword arguments (line 22)
        kwargs_492653 = {}
        # Getting the type of 'np' (line 22)
        np_492637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 22)
        array_492638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 21), np_492637, 'array')
        # Calling array(args, kwargs) (line 22)
        array_call_result_492654 = invoke(stypy.reporting.localization.Localization(__file__, 22, 21), array_492638, *[list_492639, str_492652], **kwargs_492653)
        
        # Getting the type of 'self' (line 22)
        self_492655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'data3' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_492655, 'data3', array_call_result_492654)
        
        # Assigning a BinOp to a Name (line 27):
        
        # Assigning a BinOp to a Name (line 27):
        # Getting the type of 'np' (line 27)
        np_492656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'np')
        # Obtaining the member 'pi' of a type (line 27)
        pi_492657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), np_492656, 'pi')
        int_492658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 29), 'int')
        # Applying the binary operator 'div' (line 27)
        result_div_492659 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 21), 'div', pi_492657, int_492658)
        
        # Assigning a type to the variable 'shiftangle' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'shiftangle', result_div_492659)
        
        # Assigning a BinOp to a Attribute (line 28):
        
        # Assigning a BinOp to a Attribute (line 28):
        
        # Call to array(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_492662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_492663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        int_492664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 31), list_492663, int_492664)
        # Adding element type (line 28)
        int_492665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 31), list_492663, int_492665)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 30), list_492662, list_492663)
        # Adding element type (line 28)
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_492666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        int_492667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 39), list_492666, int_492667)
        # Adding element type (line 28)
        int_492668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 39), list_492666, int_492668)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 30), list_492662, list_492666)
        # Adding element type (line 28)
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_492669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        int_492670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 47), list_492669, int_492670)
        # Adding element type (line 28)
        int_492671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 47), list_492669, int_492671)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 30), list_492662, list_492669)
        # Adding element type (line 28)
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_492672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        int_492673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 30), list_492672, int_492673)
        # Adding element type (line 29)
        int_492674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 30), list_492672, int_492674)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 30), list_492662, list_492672)
        
        str_492675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 40), 'str', 'd')
        # Processing the call keyword arguments (line 28)
        kwargs_492676 = {}
        # Getting the type of 'np' (line 28)
        np_492660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 28)
        array_492661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 21), np_492660, 'array')
        # Calling array(args, kwargs) (line 28)
        array_call_result_492677 = invoke(stypy.reporting.localization.Localization(__file__, 28, 21), array_492661, *[list_492662, str_492675], **kwargs_492676)
        
        
        # Call to sqrt(...): (line 29)
        # Processing the call arguments (line 29)
        int_492680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 55), 'int')
        # Processing the call keyword arguments (line 29)
        kwargs_492681 = {}
        # Getting the type of 'np' (line 29)
        np_492678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 47), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 29)
        sqrt_492679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 47), np_492678, 'sqrt')
        # Calling sqrt(args, kwargs) (line 29)
        sqrt_call_result_492682 = invoke(stypy.reporting.localization.Localization(__file__, 29, 47), sqrt_492679, *[int_492680], **kwargs_492681)
        
        # Applying the binary operator 'div' (line 28)
        result_div_492683 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 21), 'div', array_call_result_492677, sqrt_call_result_492682)
        
        # Getting the type of 'self' (line 28)
        self_492684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'data4' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_492684, 'data4', result_div_492683)
        
        # Assigning a BinOp to a Attribute (line 30):
        
        # Assigning a BinOp to a Attribute (line 30):
        
        # Call to array(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_492687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_492688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        
        # Call to cos(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'shiftangle' (line 30)
        shiftangle_492691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'shiftangle', False)
        # Processing the call keyword arguments (line 30)
        kwargs_492692 = {}
        # Getting the type of 'np' (line 30)
        np_492689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'np', False)
        # Obtaining the member 'cos' of a type (line 30)
        cos_492690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 32), np_492689, 'cos')
        # Calling cos(args, kwargs) (line 30)
        cos_call_result_492693 = invoke(stypy.reporting.localization.Localization(__file__, 30, 32), cos_492690, *[shiftangle_492691], **kwargs_492692)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_492688, cos_call_result_492693)
        # Adding element type (line 30)
        
        # Call to sin(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'shiftangle' (line 30)
        shiftangle_492696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 59), 'shiftangle', False)
        # Processing the call keyword arguments (line 30)
        kwargs_492697 = {}
        # Getting the type of 'np' (line 30)
        np_492694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 52), 'np', False)
        # Obtaining the member 'sin' of a type (line 30)
        sin_492695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 52), np_492694, 'sin')
        # Calling sin(args, kwargs) (line 30)
        sin_call_result_492698 = invoke(stypy.reporting.localization.Localization(__file__, 30, 52), sin_492695, *[shiftangle_492696], **kwargs_492697)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_492688, sin_call_result_492698)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 30), list_492687, list_492688)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_492699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        
        # Call to cos(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'np' (line 31)
        np_492702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 38), 'np', False)
        # Obtaining the member 'pi' of a type (line 31)
        pi_492703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 38), np_492702, 'pi')
        int_492704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 46), 'int')
        # Applying the binary operator 'div' (line 31)
        result_div_492705 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 38), 'div', pi_492703, int_492704)
        
        # Getting the type of 'shiftangle' (line 31)
        shiftangle_492706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 50), 'shiftangle', False)
        # Applying the binary operator '-' (line 31)
        result_sub_492707 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 38), '-', result_div_492705, shiftangle_492706)
        
        # Processing the call keyword arguments (line 31)
        kwargs_492708 = {}
        # Getting the type of 'np' (line 31)
        np_492700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'np', False)
        # Obtaining the member 'cos' of a type (line 31)
        cos_492701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 31), np_492700, 'cos')
        # Calling cos(args, kwargs) (line 31)
        cos_call_result_492709 = invoke(stypy.reporting.localization.Localization(__file__, 31, 31), cos_492701, *[result_sub_492707], **kwargs_492708)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 30), list_492699, cos_call_result_492709)
        # Adding element type (line 31)
        
        # Call to sin(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'np' (line 32)
        np_492712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 38), 'np', False)
        # Obtaining the member 'pi' of a type (line 32)
        pi_492713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 38), np_492712, 'pi')
        int_492714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 46), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_492715 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 38), 'div', pi_492713, int_492714)
        
        # Getting the type of 'shiftangle' (line 32)
        shiftangle_492716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 50), 'shiftangle', False)
        # Applying the binary operator '-' (line 32)
        result_sub_492717 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 38), '-', result_div_492715, shiftangle_492716)
        
        # Processing the call keyword arguments (line 32)
        kwargs_492718 = {}
        # Getting the type of 'np' (line 32)
        np_492710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'np', False)
        # Obtaining the member 'sin' of a type (line 32)
        sin_492711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 31), np_492710, 'sin')
        # Calling sin(args, kwargs) (line 32)
        sin_call_result_492719 = invoke(stypy.reporting.localization.Localization(__file__, 32, 31), sin_492711, *[result_sub_492717], **kwargs_492718)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 30), list_492699, sin_call_result_492719)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 30), list_492687, list_492699)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_492720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        
        
        # Call to cos(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'shiftangle' (line 33)
        shiftangle_492723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 39), 'shiftangle', False)
        # Processing the call keyword arguments (line 33)
        kwargs_492724 = {}
        # Getting the type of 'np' (line 33)
        np_492721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 32), 'np', False)
        # Obtaining the member 'cos' of a type (line 33)
        cos_492722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 32), np_492721, 'cos')
        # Calling cos(args, kwargs) (line 33)
        cos_call_result_492725 = invoke(stypy.reporting.localization.Localization(__file__, 33, 32), cos_492722, *[shiftangle_492723], **kwargs_492724)
        
        # Applying the 'usub' unary operator (line 33)
        result___neg___492726 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 31), 'usub', cos_call_result_492725)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 30), list_492720, result___neg___492726)
        # Adding element type (line 33)
        
        
        # Call to sin(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'shiftangle' (line 34)
        shiftangle_492729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 39), 'shiftangle', False)
        # Processing the call keyword arguments (line 34)
        kwargs_492730 = {}
        # Getting the type of 'np' (line 34)
        np_492727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'np', False)
        # Obtaining the member 'sin' of a type (line 34)
        sin_492728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 32), np_492727, 'sin')
        # Calling sin(args, kwargs) (line 34)
        sin_call_result_492731 = invoke(stypy.reporting.localization.Localization(__file__, 34, 32), sin_492728, *[shiftangle_492729], **kwargs_492730)
        
        # Applying the 'usub' unary operator (line 34)
        result___neg___492732 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 31), 'usub', sin_call_result_492731)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 30), list_492720, result___neg___492732)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 30), list_492687, list_492720)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_492733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        
        
        # Call to cos(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'np' (line 35)
        np_492736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'np', False)
        # Obtaining the member 'pi' of a type (line 35)
        pi_492737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 39), np_492736, 'pi')
        int_492738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 47), 'int')
        # Applying the binary operator 'div' (line 35)
        result_div_492739 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 39), 'div', pi_492737, int_492738)
        
        # Getting the type of 'shiftangle' (line 35)
        shiftangle_492740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 51), 'shiftangle', False)
        # Applying the binary operator '-' (line 35)
        result_sub_492741 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 39), '-', result_div_492739, shiftangle_492740)
        
        # Processing the call keyword arguments (line 35)
        kwargs_492742 = {}
        # Getting the type of 'np' (line 35)
        np_492734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'np', False)
        # Obtaining the member 'cos' of a type (line 35)
        cos_492735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 32), np_492734, 'cos')
        # Calling cos(args, kwargs) (line 35)
        cos_call_result_492743 = invoke(stypy.reporting.localization.Localization(__file__, 35, 32), cos_492735, *[result_sub_492741], **kwargs_492742)
        
        # Applying the 'usub' unary operator (line 35)
        result___neg___492744 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 31), 'usub', cos_call_result_492743)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 30), list_492733, result___neg___492744)
        # Adding element type (line 35)
        
        
        # Call to sin(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'np' (line 36)
        np_492747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 39), 'np', False)
        # Obtaining the member 'pi' of a type (line 36)
        pi_492748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 39), np_492747, 'pi')
        int_492749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 47), 'int')
        # Applying the binary operator 'div' (line 36)
        result_div_492750 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 39), 'div', pi_492748, int_492749)
        
        # Getting the type of 'shiftangle' (line 36)
        shiftangle_492751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 51), 'shiftangle', False)
        # Applying the binary operator '-' (line 36)
        result_sub_492752 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 39), '-', result_div_492750, shiftangle_492751)
        
        # Processing the call keyword arguments (line 36)
        kwargs_492753 = {}
        # Getting the type of 'np' (line 36)
        np_492745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 32), 'np', False)
        # Obtaining the member 'sin' of a type (line 36)
        sin_492746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 32), np_492745, 'sin')
        # Calling sin(args, kwargs) (line 36)
        sin_call_result_492754 = invoke(stypy.reporting.localization.Localization(__file__, 36, 32), sin_492746, *[result_sub_492752], **kwargs_492753)
        
        # Applying the 'usub' unary operator (line 36)
        result___neg___492755 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 31), 'usub', sin_call_result_492754)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 30), list_492733, result___neg___492755)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 30), list_492687, list_492733)
        
        str_492756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'str', 'd')
        # Processing the call keyword arguments (line 30)
        kwargs_492757 = {}
        # Getting the type of 'np' (line 30)
        np_492685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 30)
        array_492686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), np_492685, 'array')
        # Calling array(args, kwargs) (line 30)
        array_call_result_492758 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), array_492686, *[list_492687, str_492756], **kwargs_492757)
        
        
        # Call to sqrt(...): (line 37)
        # Processing the call arguments (line 37)
        int_492761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 45), 'int')
        # Processing the call keyword arguments (line 37)
        kwargs_492762 = {}
        # Getting the type of 'np' (line 37)
        np_492759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 37), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 37)
        sqrt_492760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 37), np_492759, 'sqrt')
        # Calling sqrt(args, kwargs) (line 37)
        sqrt_call_result_492763 = invoke(stypy.reporting.localization.Localization(__file__, 37, 37), sqrt_492760, *[int_492761], **kwargs_492762)
        
        # Applying the binary operator 'div' (line 30)
        result_div_492764 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 21), 'div', array_call_result_492758, sqrt_call_result_492763)
        
        # Getting the type of 'self' (line 30)
        self_492765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'data5' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_492765, 'data5', result_div_492764)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_492766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_492766


    @norecursion
    def test_procrustes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_procrustes'
        module_type_store = module_type_store.open_function_context('test_procrustes', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_localization', localization)
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_function_name', 'TestProcrustes.test_procrustes')
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_param_names_list', [])
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProcrustes.test_procrustes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProcrustes.test_procrustes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_procrustes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_procrustes(...)' code ##################

        
        # Assigning a Call to a Tuple (line 46):
        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        int_492767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'int')
        
        # Call to procrustes(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_492769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'self', False)
        # Obtaining the member 'data1' of a type (line 46)
        data1_492770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 37), self_492769, 'data1')
        # Getting the type of 'self' (line 46)
        self_492771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 49), 'self', False)
        # Obtaining the member 'data2' of a type (line 46)
        data2_492772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 49), self_492771, 'data2')
        # Processing the call keyword arguments (line 46)
        kwargs_492773 = {}
        # Getting the type of 'procrustes' (line 46)
        procrustes_492768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 46)
        procrustes_call_result_492774 = invoke(stypy.reporting.localization.Localization(__file__, 46, 26), procrustes_492768, *[data1_492770, data2_492772], **kwargs_492773)
        
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___492775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), procrustes_call_result_492774, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_492776 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), getitem___492775, int_492767)
        
        # Assigning a type to the variable 'tuple_var_assignment_492569' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_492569', subscript_call_result_492776)
        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        int_492777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'int')
        
        # Call to procrustes(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_492779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'self', False)
        # Obtaining the member 'data1' of a type (line 46)
        data1_492780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 37), self_492779, 'data1')
        # Getting the type of 'self' (line 46)
        self_492781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 49), 'self', False)
        # Obtaining the member 'data2' of a type (line 46)
        data2_492782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 49), self_492781, 'data2')
        # Processing the call keyword arguments (line 46)
        kwargs_492783 = {}
        # Getting the type of 'procrustes' (line 46)
        procrustes_492778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 46)
        procrustes_call_result_492784 = invoke(stypy.reporting.localization.Localization(__file__, 46, 26), procrustes_492778, *[data1_492780, data2_492782], **kwargs_492783)
        
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___492785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), procrustes_call_result_492784, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_492786 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), getitem___492785, int_492777)
        
        # Assigning a type to the variable 'tuple_var_assignment_492570' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_492570', subscript_call_result_492786)
        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        int_492787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'int')
        
        # Call to procrustes(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_492789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'self', False)
        # Obtaining the member 'data1' of a type (line 46)
        data1_492790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 37), self_492789, 'data1')
        # Getting the type of 'self' (line 46)
        self_492791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 49), 'self', False)
        # Obtaining the member 'data2' of a type (line 46)
        data2_492792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 49), self_492791, 'data2')
        # Processing the call keyword arguments (line 46)
        kwargs_492793 = {}
        # Getting the type of 'procrustes' (line 46)
        procrustes_492788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 46)
        procrustes_call_result_492794 = invoke(stypy.reporting.localization.Localization(__file__, 46, 26), procrustes_492788, *[data1_492790, data2_492792], **kwargs_492793)
        
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___492795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), procrustes_call_result_492794, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_492796 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), getitem___492795, int_492787)
        
        # Assigning a type to the variable 'tuple_var_assignment_492571' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_492571', subscript_call_result_492796)
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'tuple_var_assignment_492569' (line 46)
        tuple_var_assignment_492569_492797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_492569')
        # Assigning a type to the variable 'a' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'a', tuple_var_assignment_492569_492797)
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'tuple_var_assignment_492570' (line 46)
        tuple_var_assignment_492570_492798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_492570')
        # Assigning a type to the variable 'b' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'b', tuple_var_assignment_492570_492798)
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'tuple_var_assignment_492571' (line 46)
        tuple_var_assignment_492571_492799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_492571')
        # Assigning a type to the variable 'disparity' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'disparity', tuple_var_assignment_492571_492799)
        
        # Call to assert_allclose(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'b' (line 47)
        b_492801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'b', False)
        # Getting the type of 'a' (line 47)
        a_492802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'a', False)
        # Processing the call keyword arguments (line 47)
        kwargs_492803 = {}
        # Getting the type of 'assert_allclose' (line 47)
        assert_allclose_492800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 47)
        assert_allclose_call_result_492804 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_allclose_492800, *[b_492801, a_492802], **kwargs_492803)
        
        
        # Call to assert_almost_equal(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'disparity' (line 48)
        disparity_492806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'disparity', False)
        float_492807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 39), 'float')
        # Processing the call keyword arguments (line 48)
        kwargs_492808 = {}
        # Getting the type of 'assert_almost_equal' (line 48)
        assert_almost_equal_492805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 48)
        assert_almost_equal_call_result_492809 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_almost_equal_492805, *[disparity_492806, float_492807], **kwargs_492808)
        
        
        # Assigning a Call to a Tuple (line 51):
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_492810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'int')
        
        # Call to procrustes(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_492812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 36), 'self', False)
        # Obtaining the member 'data4' of a type (line 51)
        data4_492813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 36), self_492812, 'data4')
        # Getting the type of 'self' (line 51)
        self_492814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 48), 'self', False)
        # Obtaining the member 'data5' of a type (line 51)
        data5_492815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 48), self_492814, 'data5')
        # Processing the call keyword arguments (line 51)
        kwargs_492816 = {}
        # Getting the type of 'procrustes' (line 51)
        procrustes_492811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 51)
        procrustes_call_result_492817 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), procrustes_492811, *[data4_492813, data5_492815], **kwargs_492816)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___492818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), procrustes_call_result_492817, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_492819 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), getitem___492818, int_492810)
        
        # Assigning a type to the variable 'tuple_var_assignment_492572' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'tuple_var_assignment_492572', subscript_call_result_492819)
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_492820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'int')
        
        # Call to procrustes(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_492822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 36), 'self', False)
        # Obtaining the member 'data4' of a type (line 51)
        data4_492823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 36), self_492822, 'data4')
        # Getting the type of 'self' (line 51)
        self_492824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 48), 'self', False)
        # Obtaining the member 'data5' of a type (line 51)
        data5_492825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 48), self_492824, 'data5')
        # Processing the call keyword arguments (line 51)
        kwargs_492826 = {}
        # Getting the type of 'procrustes' (line 51)
        procrustes_492821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 51)
        procrustes_call_result_492827 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), procrustes_492821, *[data4_492823, data5_492825], **kwargs_492826)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___492828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), procrustes_call_result_492827, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_492829 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), getitem___492828, int_492820)
        
        # Assigning a type to the variable 'tuple_var_assignment_492573' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'tuple_var_assignment_492573', subscript_call_result_492829)
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_492830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'int')
        
        # Call to procrustes(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_492832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 36), 'self', False)
        # Obtaining the member 'data4' of a type (line 51)
        data4_492833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 36), self_492832, 'data4')
        # Getting the type of 'self' (line 51)
        self_492834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 48), 'self', False)
        # Obtaining the member 'data5' of a type (line 51)
        data5_492835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 48), self_492834, 'data5')
        # Processing the call keyword arguments (line 51)
        kwargs_492836 = {}
        # Getting the type of 'procrustes' (line 51)
        procrustes_492831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 51)
        procrustes_call_result_492837 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), procrustes_492831, *[data4_492833, data5_492835], **kwargs_492836)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___492838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), procrustes_call_result_492837, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_492839 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), getitem___492838, int_492830)
        
        # Assigning a type to the variable 'tuple_var_assignment_492574' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'tuple_var_assignment_492574', subscript_call_result_492839)
        
        # Assigning a Name to a Name (line 51):
        # Getting the type of 'tuple_var_assignment_492572' (line 51)
        tuple_var_assignment_492572_492840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'tuple_var_assignment_492572')
        # Assigning a type to the variable 'm4' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'm4', tuple_var_assignment_492572_492840)
        
        # Assigning a Name to a Name (line 51):
        # Getting the type of 'tuple_var_assignment_492573' (line 51)
        tuple_var_assignment_492573_492841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'tuple_var_assignment_492573')
        # Assigning a type to the variable 'm5' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'm5', tuple_var_assignment_492573_492841)
        
        # Assigning a Name to a Name (line 51):
        # Getting the type of 'tuple_var_assignment_492574' (line 51)
        tuple_var_assignment_492574_492842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'tuple_var_assignment_492574')
        # Assigning a type to the variable 'disp45' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'disp45', tuple_var_assignment_492574_492842)
        
        # Call to assert_equal(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'm4' (line 52)
        m4_492844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'm4', False)
        # Getting the type of 'self' (line 52)
        self_492845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'self', False)
        # Obtaining the member 'data4' of a type (line 52)
        data4_492846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 25), self_492845, 'data4')
        # Processing the call keyword arguments (line 52)
        kwargs_492847 = {}
        # Getting the type of 'assert_equal' (line 52)
        assert_equal_492843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 52)
        assert_equal_call_result_492848 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assert_equal_492843, *[m4_492844, data4_492846], **kwargs_492847)
        
        
        # Assigning a Call to a Tuple (line 55):
        
        # Assigning a Subscript to a Name (line 55):
        
        # Obtaining the type of the subscript
        int_492849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
        
        # Call to procrustes(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_492851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'self', False)
        # Obtaining the member 'data1' of a type (line 55)
        data1_492852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 36), self_492851, 'data1')
        # Getting the type of 'self' (line 55)
        self_492853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 48), 'self', False)
        # Obtaining the member 'data3' of a type (line 55)
        data3_492854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 48), self_492853, 'data3')
        # Processing the call keyword arguments (line 55)
        kwargs_492855 = {}
        # Getting the type of 'procrustes' (line 55)
        procrustes_492850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 55)
        procrustes_call_result_492856 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), procrustes_492850, *[data1_492852, data3_492854], **kwargs_492855)
        
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___492857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), procrustes_call_result_492856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_492858 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___492857, int_492849)
        
        # Assigning a type to the variable 'tuple_var_assignment_492575' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_492575', subscript_call_result_492858)
        
        # Assigning a Subscript to a Name (line 55):
        
        # Obtaining the type of the subscript
        int_492859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
        
        # Call to procrustes(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_492861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'self', False)
        # Obtaining the member 'data1' of a type (line 55)
        data1_492862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 36), self_492861, 'data1')
        # Getting the type of 'self' (line 55)
        self_492863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 48), 'self', False)
        # Obtaining the member 'data3' of a type (line 55)
        data3_492864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 48), self_492863, 'data3')
        # Processing the call keyword arguments (line 55)
        kwargs_492865 = {}
        # Getting the type of 'procrustes' (line 55)
        procrustes_492860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 55)
        procrustes_call_result_492866 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), procrustes_492860, *[data1_492862, data3_492864], **kwargs_492865)
        
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___492867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), procrustes_call_result_492866, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_492868 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___492867, int_492859)
        
        # Assigning a type to the variable 'tuple_var_assignment_492576' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_492576', subscript_call_result_492868)
        
        # Assigning a Subscript to a Name (line 55):
        
        # Obtaining the type of the subscript
        int_492869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
        
        # Call to procrustes(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_492871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'self', False)
        # Obtaining the member 'data1' of a type (line 55)
        data1_492872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 36), self_492871, 'data1')
        # Getting the type of 'self' (line 55)
        self_492873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 48), 'self', False)
        # Obtaining the member 'data3' of a type (line 55)
        data3_492874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 48), self_492873, 'data3')
        # Processing the call keyword arguments (line 55)
        kwargs_492875 = {}
        # Getting the type of 'procrustes' (line 55)
        procrustes_492870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 55)
        procrustes_call_result_492876 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), procrustes_492870, *[data1_492872, data3_492874], **kwargs_492875)
        
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___492877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), procrustes_call_result_492876, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_492878 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___492877, int_492869)
        
        # Assigning a type to the variable 'tuple_var_assignment_492577' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_492577', subscript_call_result_492878)
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'tuple_var_assignment_492575' (line 55)
        tuple_var_assignment_492575_492879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_492575')
        # Assigning a type to the variable 'm1' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'm1', tuple_var_assignment_492575_492879)
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'tuple_var_assignment_492576' (line 55)
        tuple_var_assignment_492576_492880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_492576')
        # Assigning a type to the variable 'm3' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'm3', tuple_var_assignment_492576_492880)
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'tuple_var_assignment_492577' (line 55)
        tuple_var_assignment_492577_492881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_492577')
        # Assigning a type to the variable 'disp13' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'disp13', tuple_var_assignment_492577_492881)
        
        # ################# End of 'test_procrustes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_procrustes' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_492882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492882)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_procrustes'
        return stypy_return_type_492882


    @norecursion
    def test_procrustes2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_procrustes2'
        module_type_store = module_type_store.open_function_context('test_procrustes2', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_localization', localization)
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_function_name', 'TestProcrustes.test_procrustes2')
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_param_names_list', [])
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProcrustes.test_procrustes2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProcrustes.test_procrustes2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_procrustes2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_procrustes2(...)' code ##################

        
        # Assigning a Call to a Tuple (line 60):
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_492883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        
        # Call to procrustes(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_492885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'self', False)
        # Obtaining the member 'data1' of a type (line 60)
        data1_492886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 36), self_492885, 'data1')
        # Getting the type of 'self' (line 60)
        self_492887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 48), 'self', False)
        # Obtaining the member 'data3' of a type (line 60)
        data3_492888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 48), self_492887, 'data3')
        # Processing the call keyword arguments (line 60)
        kwargs_492889 = {}
        # Getting the type of 'procrustes' (line 60)
        procrustes_492884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 60)
        procrustes_call_result_492890 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), procrustes_492884, *[data1_492886, data3_492888], **kwargs_492889)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___492891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), procrustes_call_result_492890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_492892 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___492891, int_492883)
        
        # Assigning a type to the variable 'tuple_var_assignment_492578' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_492578', subscript_call_result_492892)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_492893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        
        # Call to procrustes(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_492895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'self', False)
        # Obtaining the member 'data1' of a type (line 60)
        data1_492896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 36), self_492895, 'data1')
        # Getting the type of 'self' (line 60)
        self_492897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 48), 'self', False)
        # Obtaining the member 'data3' of a type (line 60)
        data3_492898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 48), self_492897, 'data3')
        # Processing the call keyword arguments (line 60)
        kwargs_492899 = {}
        # Getting the type of 'procrustes' (line 60)
        procrustes_492894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 60)
        procrustes_call_result_492900 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), procrustes_492894, *[data1_492896, data3_492898], **kwargs_492899)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___492901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), procrustes_call_result_492900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_492902 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___492901, int_492893)
        
        # Assigning a type to the variable 'tuple_var_assignment_492579' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_492579', subscript_call_result_492902)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_492903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        
        # Call to procrustes(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_492905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'self', False)
        # Obtaining the member 'data1' of a type (line 60)
        data1_492906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 36), self_492905, 'data1')
        # Getting the type of 'self' (line 60)
        self_492907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 48), 'self', False)
        # Obtaining the member 'data3' of a type (line 60)
        data3_492908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 48), self_492907, 'data3')
        # Processing the call keyword arguments (line 60)
        kwargs_492909 = {}
        # Getting the type of 'procrustes' (line 60)
        procrustes_492904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 60)
        procrustes_call_result_492910 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), procrustes_492904, *[data1_492906, data3_492908], **kwargs_492909)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___492911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), procrustes_call_result_492910, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_492912 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___492911, int_492903)
        
        # Assigning a type to the variable 'tuple_var_assignment_492580' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_492580', subscript_call_result_492912)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_492578' (line 60)
        tuple_var_assignment_492578_492913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_492578')
        # Assigning a type to the variable 'm1' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'm1', tuple_var_assignment_492578_492913)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_492579' (line 60)
        tuple_var_assignment_492579_492914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_492579')
        # Assigning a type to the variable 'm3' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'm3', tuple_var_assignment_492579_492914)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_492580' (line 60)
        tuple_var_assignment_492580_492915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_492580')
        # Assigning a type to the variable 'disp13' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'disp13', tuple_var_assignment_492580_492915)
        
        # Assigning a Call to a Tuple (line 61):
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_492916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        
        # Call to procrustes(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_492918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 40), 'self', False)
        # Obtaining the member 'data3' of a type (line 61)
        data3_492919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 40), self_492918, 'data3')
        # Getting the type of 'self' (line 61)
        self_492920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 52), 'self', False)
        # Obtaining the member 'data1' of a type (line 61)
        data1_492921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 52), self_492920, 'data1')
        # Processing the call keyword arguments (line 61)
        kwargs_492922 = {}
        # Getting the type of 'procrustes' (line 61)
        procrustes_492917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 61)
        procrustes_call_result_492923 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), procrustes_492917, *[data3_492919, data1_492921], **kwargs_492922)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___492924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), procrustes_call_result_492923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_492925 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___492924, int_492916)
        
        # Assigning a type to the variable 'tuple_var_assignment_492581' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_492581', subscript_call_result_492925)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_492926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        
        # Call to procrustes(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_492928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 40), 'self', False)
        # Obtaining the member 'data3' of a type (line 61)
        data3_492929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 40), self_492928, 'data3')
        # Getting the type of 'self' (line 61)
        self_492930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 52), 'self', False)
        # Obtaining the member 'data1' of a type (line 61)
        data1_492931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 52), self_492930, 'data1')
        # Processing the call keyword arguments (line 61)
        kwargs_492932 = {}
        # Getting the type of 'procrustes' (line 61)
        procrustes_492927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 61)
        procrustes_call_result_492933 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), procrustes_492927, *[data3_492929, data1_492931], **kwargs_492932)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___492934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), procrustes_call_result_492933, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_492935 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___492934, int_492926)
        
        # Assigning a type to the variable 'tuple_var_assignment_492582' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_492582', subscript_call_result_492935)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_492936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        
        # Call to procrustes(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_492938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 40), 'self', False)
        # Obtaining the member 'data3' of a type (line 61)
        data3_492939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 40), self_492938, 'data3')
        # Getting the type of 'self' (line 61)
        self_492940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 52), 'self', False)
        # Obtaining the member 'data1' of a type (line 61)
        data1_492941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 52), self_492940, 'data1')
        # Processing the call keyword arguments (line 61)
        kwargs_492942 = {}
        # Getting the type of 'procrustes' (line 61)
        procrustes_492937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 61)
        procrustes_call_result_492943 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), procrustes_492937, *[data3_492939, data1_492941], **kwargs_492942)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___492944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), procrustes_call_result_492943, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_492945 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___492944, int_492936)
        
        # Assigning a type to the variable 'tuple_var_assignment_492583' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_492583', subscript_call_result_492945)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'tuple_var_assignment_492581' (line 61)
        tuple_var_assignment_492581_492946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_492581')
        # Assigning a type to the variable 'm3_2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'm3_2', tuple_var_assignment_492581_492946)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'tuple_var_assignment_492582' (line 61)
        tuple_var_assignment_492582_492947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_492582')
        # Assigning a type to the variable 'm1_2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'm1_2', tuple_var_assignment_492582_492947)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'tuple_var_assignment_492583' (line 61)
        tuple_var_assignment_492583_492948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_492583')
        # Assigning a type to the variable 'disp31' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'disp31', tuple_var_assignment_492583_492948)
        
        # Call to assert_almost_equal(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'disp13' (line 62)
        disp13_492950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'disp13', False)
        # Getting the type of 'disp31' (line 62)
        disp31_492951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 36), 'disp31', False)
        # Processing the call keyword arguments (line 62)
        kwargs_492952 = {}
        # Getting the type of 'assert_almost_equal' (line 62)
        assert_almost_equal_492949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 62)
        assert_almost_equal_call_result_492953 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_almost_equal_492949, *[disp13_492950, disp31_492951], **kwargs_492952)
        
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to array(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_492956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_492957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        float_492958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), list_492957, float_492958)
        # Adding element type (line 65)
        float_492959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), list_492957, float_492959)
        # Adding element type (line 65)
        float_492960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), list_492957, float_492960)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_492956, list_492957)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_492961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        float_492962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 25), list_492961, float_492962)
        # Adding element type (line 66)
        float_492963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 25), list_492961, float_492963)
        # Adding element type (line 66)
        float_492964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 25), list_492961, float_492964)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_492956, list_492961)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_492965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        float_492966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 25), list_492965, float_492966)
        # Adding element type (line 67)
        float_492967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 25), list_492965, float_492967)
        # Adding element type (line 67)
        float_492968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 25), list_492965, float_492968)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_492956, list_492965)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_492969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        float_492970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 25), list_492969, float_492970)
        # Adding element type (line 68)
        float_492971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 25), list_492969, float_492971)
        # Adding element type (line 68)
        float_492972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 25), list_492969, float_492972)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_492956, list_492969)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_492973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        float_492974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 25), list_492973, float_492974)
        # Adding element type (line 69)
        float_492975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 25), list_492973, float_492975)
        # Adding element type (line 69)
        float_492976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 25), list_492973, float_492976)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_492956, list_492973)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_492977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        float_492978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 25), list_492977, float_492978)
        # Adding element type (line 70)
        float_492979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 25), list_492977, float_492979)
        # Adding element type (line 70)
        float_492980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 25), list_492977, float_492980)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_492956, list_492977)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_492981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        float_492982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 25), list_492981, float_492982)
        # Adding element type (line 71)
        float_492983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 25), list_492981, float_492983)
        # Adding element type (line 71)
        float_492984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 25), list_492981, float_492984)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_492956, list_492981)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_492985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        float_492986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 25), list_492985, float_492986)
        # Adding element type (line 72)
        float_492987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 25), list_492985, float_492987)
        # Adding element type (line 72)
        float_492988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 25), list_492985, float_492988)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_492956, list_492985)
        
        # Processing the call keyword arguments (line 65)
        kwargs_492989 = {}
        # Getting the type of 'np' (line 65)
        np_492954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 65)
        array_492955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), np_492954, 'array')
        # Calling array(args, kwargs) (line 65)
        array_call_result_492990 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), array_492955, *[list_492956], **kwargs_492989)
        
        # Assigning a type to the variable 'rand1' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'rand1', array_call_result_492990)
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to array(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_492993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_492994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        float_492995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 26), list_492994, float_492995)
        # Adding element type (line 74)
        float_492996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 26), list_492994, float_492996)
        # Adding element type (line 74)
        float_492997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 26), list_492994, float_492997)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 25), list_492993, list_492994)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_492998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        float_492999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 25), list_492998, float_492999)
        # Adding element type (line 75)
        float_493000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 25), list_492998, float_493000)
        # Adding element type (line 75)
        float_493001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 25), list_492998, float_493001)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 25), list_492993, list_492998)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_493002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        float_493003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 25), list_493002, float_493003)
        # Adding element type (line 76)
        float_493004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 25), list_493002, float_493004)
        # Adding element type (line 76)
        float_493005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 25), list_493002, float_493005)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 25), list_492993, list_493002)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_493006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        float_493007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_493006, float_493007)
        # Adding element type (line 77)
        float_493008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_493006, float_493008)
        # Adding element type (line 77)
        float_493009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_493006, float_493009)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 25), list_492993, list_493006)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_493010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        float_493011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 25), list_493010, float_493011)
        # Adding element type (line 78)
        float_493012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 25), list_493010, float_493012)
        # Adding element type (line 78)
        float_493013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 25), list_493010, float_493013)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 25), list_492993, list_493010)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_493014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        float_493015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 25), list_493014, float_493015)
        # Adding element type (line 79)
        float_493016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 25), list_493014, float_493016)
        # Adding element type (line 79)
        float_493017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 25), list_493014, float_493017)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 25), list_492993, list_493014)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_493018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        float_493019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 25), list_493018, float_493019)
        # Adding element type (line 80)
        float_493020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 25), list_493018, float_493020)
        # Adding element type (line 80)
        float_493021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 25), list_493018, float_493021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 25), list_492993, list_493018)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_493022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        float_493023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), list_493022, float_493023)
        # Adding element type (line 81)
        float_493024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), list_493022, float_493024)
        # Adding element type (line 81)
        float_493025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), list_493022, float_493025)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 25), list_492993, list_493022)
        
        # Processing the call keyword arguments (line 74)
        kwargs_493026 = {}
        # Getting the type of 'np' (line 74)
        np_492991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 74)
        array_492992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), np_492991, 'array')
        # Calling array(args, kwargs) (line 74)
        array_call_result_493027 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), array_492992, *[list_492993], **kwargs_493026)
        
        # Assigning a type to the variable 'rand3' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'rand3', array_call_result_493027)
        
        # Assigning a Call to a Tuple (line 82):
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_493028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        
        # Call to procrustes(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'rand1' (line 82)
        rand1_493030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'rand1', False)
        # Getting the type of 'rand3' (line 82)
        rand3_493031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'rand3', False)
        # Processing the call keyword arguments (line 82)
        kwargs_493032 = {}
        # Getting the type of 'procrustes' (line 82)
        procrustes_493029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 82)
        procrustes_call_result_493033 = invoke(stypy.reporting.localization.Localization(__file__, 82, 29), procrustes_493029, *[rand1_493030, rand3_493031], **kwargs_493032)
        
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___493034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), procrustes_call_result_493033, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_493035 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___493034, int_493028)
        
        # Assigning a type to the variable 'tuple_var_assignment_492584' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_492584', subscript_call_result_493035)
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_493036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        
        # Call to procrustes(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'rand1' (line 82)
        rand1_493038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'rand1', False)
        # Getting the type of 'rand3' (line 82)
        rand3_493039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'rand3', False)
        # Processing the call keyword arguments (line 82)
        kwargs_493040 = {}
        # Getting the type of 'procrustes' (line 82)
        procrustes_493037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 82)
        procrustes_call_result_493041 = invoke(stypy.reporting.localization.Localization(__file__, 82, 29), procrustes_493037, *[rand1_493038, rand3_493039], **kwargs_493040)
        
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___493042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), procrustes_call_result_493041, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_493043 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___493042, int_493036)
        
        # Assigning a type to the variable 'tuple_var_assignment_492585' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_492585', subscript_call_result_493043)
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_493044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        
        # Call to procrustes(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'rand1' (line 82)
        rand1_493046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'rand1', False)
        # Getting the type of 'rand3' (line 82)
        rand3_493047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'rand3', False)
        # Processing the call keyword arguments (line 82)
        kwargs_493048 = {}
        # Getting the type of 'procrustes' (line 82)
        procrustes_493045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 82)
        procrustes_call_result_493049 = invoke(stypy.reporting.localization.Localization(__file__, 82, 29), procrustes_493045, *[rand1_493046, rand3_493047], **kwargs_493048)
        
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___493050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), procrustes_call_result_493049, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_493051 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___493050, int_493044)
        
        # Assigning a type to the variable 'tuple_var_assignment_492586' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_492586', subscript_call_result_493051)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_492584' (line 82)
        tuple_var_assignment_492584_493052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_492584')
        # Assigning a type to the variable 'res1' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'res1', tuple_var_assignment_492584_493052)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_492585' (line 82)
        tuple_var_assignment_492585_493053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_492585')
        # Assigning a type to the variable 'res3' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'res3', tuple_var_assignment_492585_493053)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_492586' (line 82)
        tuple_var_assignment_492586_493054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_492586')
        # Assigning a type to the variable 'disp13' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'disp13', tuple_var_assignment_492586_493054)
        
        # Assigning a Call to a Tuple (line 83):
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_493055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'int')
        
        # Call to procrustes(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'rand3' (line 83)
        rand3_493057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 44), 'rand3', False)
        # Getting the type of 'rand1' (line 83)
        rand1_493058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 51), 'rand1', False)
        # Processing the call keyword arguments (line 83)
        kwargs_493059 = {}
        # Getting the type of 'procrustes' (line 83)
        procrustes_493056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 83)
        procrustes_call_result_493060 = invoke(stypy.reporting.localization.Localization(__file__, 83, 33), procrustes_493056, *[rand3_493057, rand1_493058], **kwargs_493059)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___493061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), procrustes_call_result_493060, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_493062 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), getitem___493061, int_493055)
        
        # Assigning a type to the variable 'tuple_var_assignment_492587' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_492587', subscript_call_result_493062)
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_493063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'int')
        
        # Call to procrustes(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'rand3' (line 83)
        rand3_493065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 44), 'rand3', False)
        # Getting the type of 'rand1' (line 83)
        rand1_493066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 51), 'rand1', False)
        # Processing the call keyword arguments (line 83)
        kwargs_493067 = {}
        # Getting the type of 'procrustes' (line 83)
        procrustes_493064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 83)
        procrustes_call_result_493068 = invoke(stypy.reporting.localization.Localization(__file__, 83, 33), procrustes_493064, *[rand3_493065, rand1_493066], **kwargs_493067)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___493069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), procrustes_call_result_493068, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_493070 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), getitem___493069, int_493063)
        
        # Assigning a type to the variable 'tuple_var_assignment_492588' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_492588', subscript_call_result_493070)
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_493071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'int')
        
        # Call to procrustes(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'rand3' (line 83)
        rand3_493073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 44), 'rand3', False)
        # Getting the type of 'rand1' (line 83)
        rand1_493074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 51), 'rand1', False)
        # Processing the call keyword arguments (line 83)
        kwargs_493075 = {}
        # Getting the type of 'procrustes' (line 83)
        procrustes_493072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'procrustes', False)
        # Calling procrustes(args, kwargs) (line 83)
        procrustes_call_result_493076 = invoke(stypy.reporting.localization.Localization(__file__, 83, 33), procrustes_493072, *[rand3_493073, rand1_493074], **kwargs_493075)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___493077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), procrustes_call_result_493076, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_493078 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), getitem___493077, int_493071)
        
        # Assigning a type to the variable 'tuple_var_assignment_492589' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_492589', subscript_call_result_493078)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'tuple_var_assignment_492587' (line 83)
        tuple_var_assignment_492587_493079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_492587')
        # Assigning a type to the variable 'res3_2' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'res3_2', tuple_var_assignment_492587_493079)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'tuple_var_assignment_492588' (line 83)
        tuple_var_assignment_492588_493080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_492588')
        # Assigning a type to the variable 'res1_2' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'res1_2', tuple_var_assignment_492588_493080)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'tuple_var_assignment_492589' (line 83)
        tuple_var_assignment_492589_493081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_492589')
        # Assigning a type to the variable 'disp31' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'disp31', tuple_var_assignment_492589_493081)
        
        # Call to assert_almost_equal(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'disp13' (line 84)
        disp13_493083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'disp13', False)
        # Getting the type of 'disp31' (line 84)
        disp31_493084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'disp31', False)
        # Processing the call keyword arguments (line 84)
        kwargs_493085 = {}
        # Getting the type of 'assert_almost_equal' (line 84)
        assert_almost_equal_493082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 84)
        assert_almost_equal_call_result_493086 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_almost_equal_493082, *[disp13_493083, disp31_493084], **kwargs_493085)
        
        
        # ################# End of 'test_procrustes2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_procrustes2' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_493087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_493087)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_procrustes2'
        return stypy_return_type_493087


    @norecursion
    def test_procrustes_shape_mismatch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_procrustes_shape_mismatch'
        module_type_store = module_type_store.open_function_context('test_procrustes_shape_mismatch', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_localization', localization)
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_function_name', 'TestProcrustes.test_procrustes_shape_mismatch')
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_param_names_list', [])
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProcrustes.test_procrustes_shape_mismatch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProcrustes.test_procrustes_shape_mismatch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_procrustes_shape_mismatch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_procrustes_shape_mismatch(...)' code ##################

        
        # Call to assert_raises(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'ValueError' (line 87)
        ValueError_493089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'ValueError', False)
        # Getting the type of 'procrustes' (line 87)
        procrustes_493090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'procrustes', False)
        
        # Call to array(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_493093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_493094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        int_493095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 32), list_493094, int_493095)
        # Adding element type (line 88)
        int_493096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 32), list_493094, int_493096)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 31), list_493093, list_493094)
        # Adding element type (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_493097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        int_493098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 40), list_493097, int_493098)
        # Adding element type (line 88)
        int_493099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 40), list_493097, int_493099)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 31), list_493093, list_493097)
        
        # Processing the call keyword arguments (line 88)
        kwargs_493100 = {}
        # Getting the type of 'np' (line 88)
        np_493091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 88)
        array_493092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 22), np_493091, 'array')
        # Calling array(args, kwargs) (line 88)
        array_call_result_493101 = invoke(stypy.reporting.localization.Localization(__file__, 88, 22), array_493092, *[list_493093], **kwargs_493100)
        
        
        # Call to array(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_493104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_493105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        int_493106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 32), list_493105, int_493106)
        # Adding element type (line 89)
        int_493107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 32), list_493105, int_493107)
        # Adding element type (line 89)
        int_493108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 32), list_493105, int_493108)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 31), list_493104, list_493105)
        # Adding element type (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_493109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        int_493110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 43), list_493109, int_493110)
        # Adding element type (line 89)
        int_493111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 43), list_493109, int_493111)
        # Adding element type (line 89)
        int_493112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 43), list_493109, int_493112)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 31), list_493104, list_493109)
        
        # Processing the call keyword arguments (line 89)
        kwargs_493113 = {}
        # Getting the type of 'np' (line 89)
        np_493102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 89)
        array_493103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), np_493102, 'array')
        # Calling array(args, kwargs) (line 89)
        array_call_result_493114 = invoke(stypy.reporting.localization.Localization(__file__, 89, 22), array_493103, *[list_493104], **kwargs_493113)
        
        # Processing the call keyword arguments (line 87)
        kwargs_493115 = {}
        # Getting the type of 'assert_raises' (line 87)
        assert_raises_493088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 87)
        assert_raises_call_result_493116 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), assert_raises_493088, *[ValueError_493089, procrustes_493090, array_call_result_493101, array_call_result_493114], **kwargs_493115)
        
        
        # ################# End of 'test_procrustes_shape_mismatch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_procrustes_shape_mismatch' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_493117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_493117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_procrustes_shape_mismatch'
        return stypy_return_type_493117


    @norecursion
    def test_procrustes_empty_rows_or_cols(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_procrustes_empty_rows_or_cols'
        module_type_store = module_type_store.open_function_context('test_procrustes_empty_rows_or_cols', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_localization', localization)
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_function_name', 'TestProcrustes.test_procrustes_empty_rows_or_cols')
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_param_names_list', [])
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProcrustes.test_procrustes_empty_rows_or_cols.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProcrustes.test_procrustes_empty_rows_or_cols', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_procrustes_empty_rows_or_cols', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_procrustes_empty_rows_or_cols(...)' code ##################

        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to array(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_493120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_493121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 25), list_493120, list_493121)
        
        # Processing the call keyword arguments (line 92)
        kwargs_493122 = {}
        # Getting the type of 'np' (line 92)
        np_493118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 92)
        array_493119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), np_493118, 'array')
        # Calling array(args, kwargs) (line 92)
        array_call_result_493123 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), array_493119, *[list_493120], **kwargs_493122)
        
        # Assigning a type to the variable 'empty' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'empty', array_call_result_493123)
        
        # Call to assert_raises(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'ValueError' (line 93)
        ValueError_493125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'ValueError', False)
        # Getting the type of 'procrustes' (line 93)
        procrustes_493126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 'procrustes', False)
        # Getting the type of 'empty' (line 93)
        empty_493127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 46), 'empty', False)
        # Getting the type of 'empty' (line 93)
        empty_493128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 53), 'empty', False)
        # Processing the call keyword arguments (line 93)
        kwargs_493129 = {}
        # Getting the type of 'assert_raises' (line 93)
        assert_raises_493124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 93)
        assert_raises_call_result_493130 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), assert_raises_493124, *[ValueError_493125, procrustes_493126, empty_493127, empty_493128], **kwargs_493129)
        
        
        # ################# End of 'test_procrustes_empty_rows_or_cols(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_procrustes_empty_rows_or_cols' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_493131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_493131)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_procrustes_empty_rows_or_cols'
        return stypy_return_type_493131


    @norecursion
    def test_procrustes_no_variation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_procrustes_no_variation'
        module_type_store = module_type_store.open_function_context('test_procrustes_no_variation', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_localization', localization)
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_function_name', 'TestProcrustes.test_procrustes_no_variation')
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_param_names_list', [])
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProcrustes.test_procrustes_no_variation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProcrustes.test_procrustes_no_variation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_procrustes_no_variation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_procrustes_no_variation(...)' code ##################

        
        # Call to assert_raises(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'ValueError' (line 96)
        ValueError_493133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'ValueError', False)
        # Getting the type of 'procrustes' (line 96)
        procrustes_493134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'procrustes', False)
        
        # Call to array(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_493137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_493138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        int_493139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 32), list_493138, int_493139)
        # Adding element type (line 97)
        int_493140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 32), list_493138, int_493140)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 31), list_493137, list_493138)
        # Adding element type (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_493141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        int_493142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 42), list_493141, int_493142)
        # Adding element type (line 97)
        int_493143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 42), list_493141, int_493143)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 31), list_493137, list_493141)
        
        # Processing the call keyword arguments (line 97)
        kwargs_493144 = {}
        # Getting the type of 'np' (line 97)
        np_493135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 97)
        array_493136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 22), np_493135, 'array')
        # Calling array(args, kwargs) (line 97)
        array_call_result_493145 = invoke(stypy.reporting.localization.Localization(__file__, 97, 22), array_493136, *[list_493137], **kwargs_493144)
        
        
        # Call to array(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_493148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_493149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        int_493150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 32), list_493149, int_493150)
        # Adding element type (line 98)
        int_493151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 32), list_493149, int_493151)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 31), list_493148, list_493149)
        # Adding element type (line 98)
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_493152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        int_493153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 42), list_493152, int_493153)
        # Adding element type (line 98)
        int_493154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 42), list_493152, int_493154)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 31), list_493148, list_493152)
        
        # Processing the call keyword arguments (line 98)
        kwargs_493155 = {}
        # Getting the type of 'np' (line 98)
        np_493146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 98)
        array_493147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 22), np_493146, 'array')
        # Calling array(args, kwargs) (line 98)
        array_call_result_493156 = invoke(stypy.reporting.localization.Localization(__file__, 98, 22), array_493147, *[list_493148], **kwargs_493155)
        
        # Processing the call keyword arguments (line 96)
        kwargs_493157 = {}
        # Getting the type of 'assert_raises' (line 96)
        assert_raises_493132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 96)
        assert_raises_call_result_493158 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assert_raises_493132, *[ValueError_493133, procrustes_493134, array_call_result_493145, array_call_result_493156], **kwargs_493157)
        
        
        # ################# End of 'test_procrustes_no_variation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_procrustes_no_variation' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_493159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_493159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_procrustes_no_variation'
        return stypy_return_type_493159


    @norecursion
    def test_procrustes_bad_number_of_dimensions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_procrustes_bad_number_of_dimensions'
        module_type_store = module_type_store.open_function_context('test_procrustes_bad_number_of_dimensions', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_localization', localization)
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_function_name', 'TestProcrustes.test_procrustes_bad_number_of_dimensions')
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_param_names_list', [])
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProcrustes.test_procrustes_bad_number_of_dimensions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProcrustes.test_procrustes_bad_number_of_dimensions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_procrustes_bad_number_of_dimensions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_procrustes_bad_number_of_dimensions(...)' code ##################

        
        # Call to assert_raises(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'ValueError' (line 102)
        ValueError_493161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'ValueError', False)
        # Getting the type of 'procrustes' (line 102)
        procrustes_493162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'procrustes', False)
        
        # Call to array(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_493165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        int_493166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 31), list_493165, int_493166)
        # Adding element type (line 103)
        int_493167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 31), list_493165, int_493167)
        # Adding element type (line 103)
        int_493168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 31), list_493165, int_493168)
        # Adding element type (line 103)
        int_493169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 31), list_493165, int_493169)
        # Adding element type (line 103)
        int_493170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 31), list_493165, int_493170)
        # Adding element type (line 103)
        int_493171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 31), list_493165, int_493171)
        
        # Processing the call keyword arguments (line 103)
        kwargs_493172 = {}
        # Getting the type of 'np' (line 103)
        np_493163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 103)
        array_493164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 22), np_493163, 'array')
        # Calling array(args, kwargs) (line 103)
        array_call_result_493173 = invoke(stypy.reporting.localization.Localization(__file__, 103, 22), array_493164, *[list_493165], **kwargs_493172)
        
        
        # Call to array(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_493176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_493177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        int_493178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 32), list_493177, int_493178)
        # Adding element type (line 104)
        int_493179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 32), list_493177, int_493179)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 31), list_493176, list_493177)
        # Adding element type (line 104)
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_493180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        int_493181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 40), list_493180, int_493181)
        # Adding element type (line 104)
        int_493182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 40), list_493180, int_493182)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 31), list_493176, list_493180)
        
        # Processing the call keyword arguments (line 104)
        kwargs_493183 = {}
        # Getting the type of 'np' (line 104)
        np_493174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 104)
        array_493175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 22), np_493174, 'array')
        # Calling array(args, kwargs) (line 104)
        array_call_result_493184 = invoke(stypy.reporting.localization.Localization(__file__, 104, 22), array_493175, *[list_493176], **kwargs_493183)
        
        # Processing the call keyword arguments (line 102)
        kwargs_493185 = {}
        # Getting the type of 'assert_raises' (line 102)
        assert_raises_493160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 102)
        assert_raises_call_result_493186 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_raises_493160, *[ValueError_493161, procrustes_493162, array_call_result_493173, array_call_result_493184], **kwargs_493185)
        
        
        # Call to assert_raises(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'ValueError' (line 107)
        ValueError_493188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'ValueError', False)
        # Getting the type of 'procrustes' (line 107)
        procrustes_493189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'procrustes', False)
        
        # Call to array(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_493192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        int_493193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), list_493192, int_493193)
        # Adding element type (line 108)
        int_493194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), list_493192, int_493194)
        # Adding element type (line 108)
        int_493195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), list_493192, int_493195)
        # Adding element type (line 108)
        int_493196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), list_493192, int_493196)
        # Adding element type (line 108)
        int_493197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), list_493192, int_493197)
        # Adding element type (line 108)
        int_493198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), list_493192, int_493198)
        
        # Processing the call keyword arguments (line 108)
        kwargs_493199 = {}
        # Getting the type of 'np' (line 108)
        np_493190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 108)
        array_493191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 22), np_493190, 'array')
        # Calling array(args, kwargs) (line 108)
        array_call_result_493200 = invoke(stypy.reporting.localization.Localization(__file__, 108, 22), array_493191, *[list_493192], **kwargs_493199)
        
        
        # Call to array(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_493203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_493204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 31), list_493203, int_493204)
        # Adding element type (line 109)
        int_493205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 31), list_493203, int_493205)
        # Adding element type (line 109)
        int_493206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 31), list_493203, int_493206)
        # Adding element type (line 109)
        int_493207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 31), list_493203, int_493207)
        # Adding element type (line 109)
        int_493208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 31), list_493203, int_493208)
        # Adding element type (line 109)
        int_493209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 31), list_493203, int_493209)
        
        # Processing the call keyword arguments (line 109)
        kwargs_493210 = {}
        # Getting the type of 'np' (line 109)
        np_493201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 109)
        array_493202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 22), np_493201, 'array')
        # Calling array(args, kwargs) (line 109)
        array_call_result_493211 = invoke(stypy.reporting.localization.Localization(__file__, 109, 22), array_493202, *[list_493203], **kwargs_493210)
        
        # Processing the call keyword arguments (line 107)
        kwargs_493212 = {}
        # Getting the type of 'assert_raises' (line 107)
        assert_raises_493187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 107)
        assert_raises_call_result_493213 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert_raises_493187, *[ValueError_493188, procrustes_493189, array_call_result_493200, array_call_result_493211], **kwargs_493212)
        
        
        # Call to assert_raises(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'ValueError' (line 112)
        ValueError_493215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'ValueError', False)
        # Getting the type of 'procrustes' (line 112)
        procrustes_493216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'procrustes', False)
        
        # Call to array(...): (line 112)
        # Processing the call arguments (line 112)
        int_493219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 55), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_493220 = {}
        # Getting the type of 'np' (line 112)
        np_493217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'np', False)
        # Obtaining the member 'array' of a type (line 112)
        array_493218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 46), np_493217, 'array')
        # Calling array(args, kwargs) (line 112)
        array_call_result_493221 = invoke(stypy.reporting.localization.Localization(__file__, 112, 46), array_493218, *[int_493219], **kwargs_493220)
        
        
        # Call to array(...): (line 112)
        # Processing the call arguments (line 112)
        int_493224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 68), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_493225 = {}
        # Getting the type of 'np' (line 112)
        np_493222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 59), 'np', False)
        # Obtaining the member 'array' of a type (line 112)
        array_493223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 59), np_493222, 'array')
        # Calling array(args, kwargs) (line 112)
        array_call_result_493226 = invoke(stypy.reporting.localization.Localization(__file__, 112, 59), array_493223, *[int_493224], **kwargs_493225)
        
        # Processing the call keyword arguments (line 112)
        kwargs_493227 = {}
        # Getting the type of 'assert_raises' (line 112)
        assert_raises_493214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 112)
        assert_raises_call_result_493228 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert_raises_493214, *[ValueError_493215, procrustes_493216, array_call_result_493221, array_call_result_493226], **kwargs_493227)
        
        
        # Call to assert_raises(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'ValueError' (line 115)
        ValueError_493230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'ValueError', False)
        # Getting the type of 'procrustes' (line 115)
        procrustes_493231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), 'procrustes', False)
        
        # Call to array(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_493234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_493235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_493236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        int_493237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 33), list_493236, int_493237)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 32), list_493235, list_493236)
        # Adding element type (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_493238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        int_493239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 39), list_493238, int_493239)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 32), list_493235, list_493238)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 31), list_493234, list_493235)
        
        # Processing the call keyword arguments (line 116)
        kwargs_493240 = {}
        # Getting the type of 'np' (line 116)
        np_493232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 116)
        array_493233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 22), np_493232, 'array')
        # Calling array(args, kwargs) (line 116)
        array_call_result_493241 = invoke(stypy.reporting.localization.Localization(__file__, 116, 22), array_493233, *[list_493234], **kwargs_493240)
        
        
        # Call to array(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_493244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_493245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_493246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        int_493247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 33), list_493246, int_493247)
        # Adding element type (line 117)
        int_493248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 33), list_493246, int_493248)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 32), list_493245, list_493246)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 31), list_493244, list_493245)
        
        # Processing the call keyword arguments (line 117)
        kwargs_493249 = {}
        # Getting the type of 'np' (line 117)
        np_493242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 117)
        array_493243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 22), np_493242, 'array')
        # Calling array(args, kwargs) (line 117)
        array_call_result_493250 = invoke(stypy.reporting.localization.Localization(__file__, 117, 22), array_493243, *[list_493244], **kwargs_493249)
        
        # Processing the call keyword arguments (line 115)
        kwargs_493251 = {}
        # Getting the type of 'assert_raises' (line 115)
        assert_raises_493229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 115)
        assert_raises_call_result_493252 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), assert_raises_493229, *[ValueError_493230, procrustes_493231, array_call_result_493241, array_call_result_493250], **kwargs_493251)
        
        
        # ################# End of 'test_procrustes_bad_number_of_dimensions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_procrustes_bad_number_of_dimensions' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_493253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_493253)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_procrustes_bad_number_of_dimensions'
        return stypy_return_type_493253


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProcrustes.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestProcrustes' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'TestProcrustes', TestProcrustes)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
