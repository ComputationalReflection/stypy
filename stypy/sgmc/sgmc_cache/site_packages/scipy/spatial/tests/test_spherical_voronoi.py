
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import print_function
2: import numpy as np
3: import itertools
4: from numpy.testing import (assert_equal,
5:                            assert_almost_equal,
6:                            assert_array_equal,
7:                            assert_array_almost_equal)
8: from pytest import raises as assert_raises
9: from scipy.spatial import SphericalVoronoi, distance
10: from scipy.spatial import _spherical_voronoi as spherical_voronoi
11: 
12: 
13: class TestCircumcenters(object):
14: 
15:     def test_circumcenters(self):
16:         tetrahedrons = np.array([
17:             [[1, 2, 3],
18:              [-1.1, -2.1, -3.1],
19:              [-1.2, 2.2, 3.2],
20:              [-1.3, -2.3, 3.3]],
21:             [[10, 20, 30],
22:              [-10.1, -20.1, -30.1],
23:              [-10.2, 20.2, 30.2],
24:              [-10.3, -20.3, 30.3]]
25:         ])
26: 
27:         result = spherical_voronoi.calc_circumcenters(tetrahedrons)
28: 
29:         expected = [
30:             [-0.5680861153262529, -0.133279590288315, 0.1843323216995444],
31:             [-0.5965330784014926, -0.1480377040397778, 0.1981967854886021]
32:         ]
33: 
34:         assert_array_almost_equal(result, expected)
35: 
36: 
37: class TestProjectToSphere(object):
38: 
39:     def test_unit_sphere(self):
40:         points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
41:         center = np.array([0, 0, 0])
42:         radius = 1
43:         projected = spherical_voronoi.project_to_sphere(points, center, radius)
44:         assert_array_almost_equal(points, projected)
45: 
46:     def test_scaled_points(self):
47:         points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
48:         center = np.array([0, 0, 0])
49:         radius = 1
50:         scaled = points * 2
51:         projected = spherical_voronoi.project_to_sphere(scaled, center, radius)
52:         assert_array_almost_equal(points, projected)
53: 
54:     def test_translated_sphere(self):
55:         points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
56:         center = np.array([1, 2, 3])
57:         translated = points + center
58:         radius = 1
59:         projected = spherical_voronoi.project_to_sphere(translated, center,
60:                                                         radius)
61:         assert_array_almost_equal(translated, projected)
62: 
63: 
64: class TestSphericalVoronoi(object):
65: 
66:     def setup_method(self):
67:         self.points = np.array([
68:             [-0.78928481, -0.16341094, 0.59188373],
69:             [-0.66839141, 0.73309634, 0.12578818],
70:             [0.32535778, -0.92476944, -0.19734181],
71:             [-0.90177102, -0.03785291, -0.43055335],
72:             [0.71781344, 0.68428936, 0.12842096],
73:             [-0.96064876, 0.23492353, -0.14820556],
74:             [0.73181537, -0.22025898, -0.6449281],
75:             [0.79979205, 0.54555747, 0.25039913]]
76:         )
77: 
78:     def test_constructor(self):
79:         center = np.array([1, 2, 3])
80:         radius = 2
81:         s1 = SphericalVoronoi(self.points)
82:         # user input checks in SphericalVoronoi now require
83:         # the radius / center to match the generators so adjust
84:         # accordingly here
85:         s2 = SphericalVoronoi(self.points * radius, radius)
86:         s3 = SphericalVoronoi(self.points + center, None, center)
87:         s4 = SphericalVoronoi(self.points * radius + center, radius, center)
88:         assert_array_equal(s1.center, np.array([0, 0, 0]))
89:         assert_equal(s1.radius, 1)
90:         assert_array_equal(s2.center, np.array([0, 0, 0]))
91:         assert_equal(s2.radius, 2)
92:         assert_array_equal(s3.center, center)
93:         assert_equal(s3.radius, 1)
94:         assert_array_equal(s4.center, center)
95:         assert_equal(s4.radius, radius)
96: 
97:     def test_vertices_regions_translation_invariance(self):
98:         sv_origin = SphericalVoronoi(self.points)
99:         center = np.array([1, 1, 1])
100:         sv_translated = SphericalVoronoi(self.points + center, None, center)
101:         assert_array_equal(sv_origin.regions, sv_translated.regions)
102:         assert_array_almost_equal(sv_origin.vertices + center,
103:                                   sv_translated.vertices)
104: 
105:     def test_vertices_regions_scaling_invariance(self):
106:         sv_unit = SphericalVoronoi(self.points)
107:         sv_scaled = SphericalVoronoi(self.points * 2, 2)
108:         assert_array_equal(sv_unit.regions, sv_scaled.regions)
109:         assert_array_almost_equal(sv_unit.vertices * 2,
110:                                   sv_scaled.vertices)
111: 
112:     def test_sort_vertices_of_regions(self):
113:         sv = SphericalVoronoi(self.points)
114:         unsorted_regions = sv.regions
115:         sv.sort_vertices_of_regions()
116:         assert_array_equal(sorted(sv.regions), sorted(unsorted_regions))
117: 
118:     def test_sort_vertices_of_regions_flattened(self):
119:         expected = sorted([[0, 6, 5, 2, 3], [2, 3, 10, 11, 8, 7], [0, 6, 4, 1], [4, 8,
120:             7, 5, 6], [9, 11, 10], [2, 7, 5], [1, 4, 8, 11, 9], [0, 3, 10, 9,
121:                 1]])
122:         expected = list(itertools.chain(*sorted(expected)))
123:         sv = SphericalVoronoi(self.points)
124:         sv.sort_vertices_of_regions()
125:         actual = list(itertools.chain(*sorted(sv.regions)))
126:         assert_array_equal(actual, expected)
127: 
128:     def test_num_vertices(self):
129:         # for any n >= 3, a spherical Voronoi diagram has 2n - 4
130:         # vertices; this is a direct consequence of Euler's formula
131:         # as explained by Dinis and Mamede (2010) Proceedings of the
132:         # 2010 International Symposium on Voronoi Diagrams in Science
133:         # and Engineering
134:         sv = SphericalVoronoi(self.points)
135:         expected = self.points.shape[0] * 2 - 4
136:         actual = sv.vertices.shape[0]
137:         assert_equal(actual, expected)
138: 
139:     def test_voronoi_circles(self):
140:         sv = spherical_voronoi.SphericalVoronoi(self.points)
141:         for vertex in sv.vertices:
142:             distances = distance.cdist(sv.points,np.array([vertex]))
143:             closest = np.array(sorted(distances)[0:3])
144:             assert_almost_equal(closest[0], closest[1], 7, str(vertex))
145:             assert_almost_equal(closest[0], closest[2], 7, str(vertex))
146: 
147:     def test_duplicate_point_handling(self):
148:         # an exception should be raised for degenerate generators
149:         # related to Issue# 7046
150:         self.degenerate = np.concatenate((self.points, self.points))
151:         with assert_raises(ValueError):
152:             sv = spherical_voronoi.SphericalVoronoi(self.degenerate)
153: 
154:     def test_incorrect_radius_handling(self):
155:         # an exception should be raised if the radius provided
156:         # cannot possibly match the input generators
157:         with assert_raises(ValueError):
158:             sv = spherical_voronoi.SphericalVoronoi(self.points,
159:                                                     radius=0.98)
160: 
161:     def test_incorrect_center_handling(self):
162:         # an exception should be raised if the center provided
163:         # cannot possibly match the input generators
164:         with assert_raises(ValueError):
165:             sv = spherical_voronoi.SphericalVoronoi(self.points,
166:                                                     center=[0.1,0,0])
167: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_491714 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_491714) is not StypyTypeError):

    if (import_491714 != 'pyd_module'):
        __import__(import_491714)
        sys_modules_491715 = sys.modules[import_491714]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_491715.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_491714)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import itertools' statement (line 3)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_491716 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_491716) is not StypyTypeError):

    if (import_491716 != 'pyd_module'):
        __import__(import_491716)
        sys_modules_491717 = sys.modules[import_491716]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_491717.module_type_store, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_array_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_491717, sys_modules_491717.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_array_equal', 'assert_array_almost_equal'], [assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_491716)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_491718 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_491718) is not StypyTypeError):

    if (import_491718 != 'pyd_module'):
        __import__(import_491718)
        sys_modules_491719 = sys.modules[import_491718]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_491719.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_491719, sys_modules_491719.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_491718)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.spatial import SphericalVoronoi, distance' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_491720 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.spatial')

if (type(import_491720) is not StypyTypeError):

    if (import_491720 != 'pyd_module'):
        __import__(import_491720)
        sys_modules_491721 = sys.modules[import_491720]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.spatial', sys_modules_491721.module_type_store, module_type_store, ['SphericalVoronoi', 'distance'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_491721, sys_modules_491721.module_type_store, module_type_store)
    else:
        from scipy.spatial import SphericalVoronoi, distance

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.spatial', None, module_type_store, ['SphericalVoronoi', 'distance'], [SphericalVoronoi, distance])

else:
    # Assigning a type to the variable 'scipy.spatial' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.spatial', import_491720)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.spatial import spherical_voronoi' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_491722 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.spatial')

if (type(import_491722) is not StypyTypeError):

    if (import_491722 != 'pyd_module'):
        __import__(import_491722)
        sys_modules_491723 = sys.modules[import_491722]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.spatial', sys_modules_491723.module_type_store, module_type_store, ['_spherical_voronoi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_491723, sys_modules_491723.module_type_store, module_type_store)
    else:
        from scipy.spatial import _spherical_voronoi as spherical_voronoi

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.spatial', None, module_type_store, ['_spherical_voronoi'], [spherical_voronoi])

else:
    # Assigning a type to the variable 'scipy.spatial' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.spatial', import_491722)

# Adding an alias
module_type_store.add_alias('spherical_voronoi', '_spherical_voronoi')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

# Declaration of the 'TestCircumcenters' class

class TestCircumcenters(object, ):

    @norecursion
    def test_circumcenters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_circumcenters'
        module_type_store = module_type_store.open_function_context('test_circumcenters', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_localization', localization)
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_function_name', 'TestCircumcenters.test_circumcenters')
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_param_names_list', [])
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCircumcenters.test_circumcenters.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCircumcenters.test_circumcenters', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_circumcenters', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_circumcenters(...)' code ##################

        
        # Assigning a Call to a Name (line 16):
        
        # Call to array(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_491726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        # Adding element type (line 16)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_491727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_491728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_491729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_491728, int_491729)
        # Adding element type (line 17)
        int_491730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_491728, int_491730)
        # Adding element type (line 17)
        int_491731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_491728, int_491731)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), list_491727, list_491728)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_491732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        # Adding element type (line 18)
        float_491733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_491732, float_491733)
        # Adding element type (line 18)
        float_491734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_491732, float_491734)
        # Adding element type (line 18)
        float_491735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_491732, float_491735)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), list_491727, list_491732)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_491736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        float_491737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 13), list_491736, float_491737)
        # Adding element type (line 19)
        float_491738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 13), list_491736, float_491738)
        # Adding element type (line 19)
        float_491739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 13), list_491736, float_491739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), list_491727, list_491736)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_491740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        float_491741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 13), list_491740, float_491741)
        # Adding element type (line 20)
        float_491742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 13), list_491740, float_491742)
        # Adding element type (line 20)
        float_491743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 13), list_491740, float_491743)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), list_491727, list_491740)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 32), list_491726, list_491727)
        # Adding element type (line 16)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_491744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_491745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        int_491746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 13), list_491745, int_491746)
        # Adding element type (line 21)
        int_491747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 13), list_491745, int_491747)
        # Adding element type (line 21)
        int_491748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 13), list_491745, int_491748)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_491744, list_491745)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_491749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        float_491750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_491749, float_491750)
        # Adding element type (line 22)
        float_491751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_491749, float_491751)
        # Adding element type (line 22)
        float_491752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_491749, float_491752)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_491744, list_491749)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_491753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        float_491754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), list_491753, float_491754)
        # Adding element type (line 23)
        float_491755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), list_491753, float_491755)
        # Adding element type (line 23)
        float_491756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), list_491753, float_491756)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_491744, list_491753)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_491757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        float_491758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_491757, float_491758)
        # Adding element type (line 24)
        float_491759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_491757, float_491759)
        # Adding element type (line 24)
        float_491760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_491757, float_491760)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_491744, list_491757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 32), list_491726, list_491744)
        
        # Processing the call keyword arguments (line 16)
        kwargs_491761 = {}
        # Getting the type of 'np' (line 16)
        np_491724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 16)
        array_491725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 23), np_491724, 'array')
        # Calling array(args, kwargs) (line 16)
        array_call_result_491762 = invoke(stypy.reporting.localization.Localization(__file__, 16, 23), array_491725, *[list_491726], **kwargs_491761)
        
        # Assigning a type to the variable 'tetrahedrons' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tetrahedrons', array_call_result_491762)
        
        # Assigning a Call to a Name (line 27):
        
        # Call to calc_circumcenters(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'tetrahedrons' (line 27)
        tetrahedrons_491765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 54), 'tetrahedrons', False)
        # Processing the call keyword arguments (line 27)
        kwargs_491766 = {}
        # Getting the type of 'spherical_voronoi' (line 27)
        spherical_voronoi_491763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'spherical_voronoi', False)
        # Obtaining the member 'calc_circumcenters' of a type (line 27)
        calc_circumcenters_491764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 17), spherical_voronoi_491763, 'calc_circumcenters')
        # Calling calc_circumcenters(args, kwargs) (line 27)
        calc_circumcenters_call_result_491767 = invoke(stypy.reporting.localization.Localization(__file__, 27, 17), calc_circumcenters_491764, *[tetrahedrons_491765], **kwargs_491766)
        
        # Assigning a type to the variable 'result' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'result', calc_circumcenters_call_result_491767)
        
        # Assigning a List to a Name (line 29):
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_491768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_491769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        float_491770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_491769, float_491770)
        # Adding element type (line 30)
        float_491771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_491769, float_491771)
        # Adding element type (line 30)
        float_491772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_491769, float_491772)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 19), list_491768, list_491769)
        # Adding element type (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_491773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        float_491774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_491773, float_491774)
        # Adding element type (line 31)
        float_491775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_491773, float_491775)
        # Adding element type (line 31)
        float_491776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_491773, float_491776)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 19), list_491768, list_491773)
        
        # Assigning a type to the variable 'expected' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'expected', list_491768)
        
        # Call to assert_array_almost_equal(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'result' (line 34)
        result_491778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'result', False)
        # Getting the type of 'expected' (line 34)
        expected_491779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'expected', False)
        # Processing the call keyword arguments (line 34)
        kwargs_491780 = {}
        # Getting the type of 'assert_array_almost_equal' (line 34)
        assert_array_almost_equal_491777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 34)
        assert_array_almost_equal_call_result_491781 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert_array_almost_equal_491777, *[result_491778, expected_491779], **kwargs_491780)
        
        
        # ################# End of 'test_circumcenters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_circumcenters' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_491782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_491782)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_circumcenters'
        return stypy_return_type_491782


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCircumcenters.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCircumcenters' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'TestCircumcenters', TestCircumcenters)
# Declaration of the 'TestProjectToSphere' class

class TestProjectToSphere(object, ):

    @norecursion
    def test_unit_sphere(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_unit_sphere'
        module_type_store = module_type_store.open_function_context('test_unit_sphere', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_localization', localization)
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_function_name', 'TestProjectToSphere.test_unit_sphere')
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_param_names_list', [])
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProjectToSphere.test_unit_sphere.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProjectToSphere.test_unit_sphere', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_unit_sphere', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_unit_sphere(...)' code ##################

        
        # Assigning a Call to a Name (line 40):
        
        # Call to array(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_491785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_491786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        int_491787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 27), list_491786, int_491787)
        # Adding element type (line 40)
        int_491788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 27), list_491786, int_491788)
        # Adding element type (line 40)
        int_491789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 27), list_491786, int_491789)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_491785, list_491786)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_491790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        int_491791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 38), list_491790, int_491791)
        # Adding element type (line 40)
        int_491792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 38), list_491790, int_491792)
        # Adding element type (line 40)
        int_491793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 38), list_491790, int_491793)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_491785, list_491790)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_491794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        int_491795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 49), list_491794, int_491795)
        # Adding element type (line 40)
        int_491796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 49), list_491794, int_491796)
        # Adding element type (line 40)
        int_491797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 49), list_491794, int_491797)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_491785, list_491794)
        
        # Processing the call keyword arguments (line 40)
        kwargs_491798 = {}
        # Getting the type of 'np' (line 40)
        np_491783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 40)
        array_491784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), np_491783, 'array')
        # Calling array(args, kwargs) (line 40)
        array_call_result_491799 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), array_491784, *[list_491785], **kwargs_491798)
        
        # Assigning a type to the variable 'points' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'points', array_call_result_491799)
        
        # Assigning a Call to a Name (line 41):
        
        # Call to array(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_491802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        int_491803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), list_491802, int_491803)
        # Adding element type (line 41)
        int_491804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), list_491802, int_491804)
        # Adding element type (line 41)
        int_491805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), list_491802, int_491805)
        
        # Processing the call keyword arguments (line 41)
        kwargs_491806 = {}
        # Getting the type of 'np' (line 41)
        np_491800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 41)
        array_491801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), np_491800, 'array')
        # Calling array(args, kwargs) (line 41)
        array_call_result_491807 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), array_491801, *[list_491802], **kwargs_491806)
        
        # Assigning a type to the variable 'center' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'center', array_call_result_491807)
        
        # Assigning a Num to a Name (line 42):
        int_491808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'int')
        # Assigning a type to the variable 'radius' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'radius', int_491808)
        
        # Assigning a Call to a Name (line 43):
        
        # Call to project_to_sphere(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'points' (line 43)
        points_491811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 56), 'points', False)
        # Getting the type of 'center' (line 43)
        center_491812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 64), 'center', False)
        # Getting the type of 'radius' (line 43)
        radius_491813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 72), 'radius', False)
        # Processing the call keyword arguments (line 43)
        kwargs_491814 = {}
        # Getting the type of 'spherical_voronoi' (line 43)
        spherical_voronoi_491809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'spherical_voronoi', False)
        # Obtaining the member 'project_to_sphere' of a type (line 43)
        project_to_sphere_491810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 20), spherical_voronoi_491809, 'project_to_sphere')
        # Calling project_to_sphere(args, kwargs) (line 43)
        project_to_sphere_call_result_491815 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), project_to_sphere_491810, *[points_491811, center_491812, radius_491813], **kwargs_491814)
        
        # Assigning a type to the variable 'projected' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'projected', project_to_sphere_call_result_491815)
        
        # Call to assert_array_almost_equal(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'points' (line 44)
        points_491817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'points', False)
        # Getting the type of 'projected' (line 44)
        projected_491818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 42), 'projected', False)
        # Processing the call keyword arguments (line 44)
        kwargs_491819 = {}
        # Getting the type of 'assert_array_almost_equal' (line 44)
        assert_array_almost_equal_491816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 44)
        assert_array_almost_equal_call_result_491820 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), assert_array_almost_equal_491816, *[points_491817, projected_491818], **kwargs_491819)
        
        
        # ################# End of 'test_unit_sphere(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_unit_sphere' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_491821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_491821)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_unit_sphere'
        return stypy_return_type_491821


    @norecursion
    def test_scaled_points(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scaled_points'
        module_type_store = module_type_store.open_function_context('test_scaled_points', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_localization', localization)
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_function_name', 'TestProjectToSphere.test_scaled_points')
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_param_names_list', [])
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProjectToSphere.test_scaled_points.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProjectToSphere.test_scaled_points', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scaled_points', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scaled_points(...)' code ##################

        
        # Assigning a Call to a Name (line 47):
        
        # Call to array(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_491824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_491825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        int_491826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 27), list_491825, int_491826)
        # Adding element type (line 47)
        int_491827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 27), list_491825, int_491827)
        # Adding element type (line 47)
        int_491828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 27), list_491825, int_491828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 26), list_491824, list_491825)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_491829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        int_491830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 38), list_491829, int_491830)
        # Adding element type (line 47)
        int_491831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 38), list_491829, int_491831)
        # Adding element type (line 47)
        int_491832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 38), list_491829, int_491832)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 26), list_491824, list_491829)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_491833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        int_491834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 49), list_491833, int_491834)
        # Adding element type (line 47)
        int_491835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 49), list_491833, int_491835)
        # Adding element type (line 47)
        int_491836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 49), list_491833, int_491836)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 26), list_491824, list_491833)
        
        # Processing the call keyword arguments (line 47)
        kwargs_491837 = {}
        # Getting the type of 'np' (line 47)
        np_491822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 47)
        array_491823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), np_491822, 'array')
        # Calling array(args, kwargs) (line 47)
        array_call_result_491838 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), array_491823, *[list_491824], **kwargs_491837)
        
        # Assigning a type to the variable 'points' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'points', array_call_result_491838)
        
        # Assigning a Call to a Name (line 48):
        
        # Call to array(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_491841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        int_491842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 26), list_491841, int_491842)
        # Adding element type (line 48)
        int_491843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 26), list_491841, int_491843)
        # Adding element type (line 48)
        int_491844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 26), list_491841, int_491844)
        
        # Processing the call keyword arguments (line 48)
        kwargs_491845 = {}
        # Getting the type of 'np' (line 48)
        np_491839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 48)
        array_491840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 17), np_491839, 'array')
        # Calling array(args, kwargs) (line 48)
        array_call_result_491846 = invoke(stypy.reporting.localization.Localization(__file__, 48, 17), array_491840, *[list_491841], **kwargs_491845)
        
        # Assigning a type to the variable 'center' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'center', array_call_result_491846)
        
        # Assigning a Num to a Name (line 49):
        int_491847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'int')
        # Assigning a type to the variable 'radius' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'radius', int_491847)
        
        # Assigning a BinOp to a Name (line 50):
        # Getting the type of 'points' (line 50)
        points_491848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'points')
        int_491849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 26), 'int')
        # Applying the binary operator '*' (line 50)
        result_mul_491850 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 17), '*', points_491848, int_491849)
        
        # Assigning a type to the variable 'scaled' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'scaled', result_mul_491850)
        
        # Assigning a Call to a Name (line 51):
        
        # Call to project_to_sphere(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'scaled' (line 51)
        scaled_491853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 56), 'scaled', False)
        # Getting the type of 'center' (line 51)
        center_491854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 64), 'center', False)
        # Getting the type of 'radius' (line 51)
        radius_491855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 72), 'radius', False)
        # Processing the call keyword arguments (line 51)
        kwargs_491856 = {}
        # Getting the type of 'spherical_voronoi' (line 51)
        spherical_voronoi_491851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'spherical_voronoi', False)
        # Obtaining the member 'project_to_sphere' of a type (line 51)
        project_to_sphere_491852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), spherical_voronoi_491851, 'project_to_sphere')
        # Calling project_to_sphere(args, kwargs) (line 51)
        project_to_sphere_call_result_491857 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), project_to_sphere_491852, *[scaled_491853, center_491854, radius_491855], **kwargs_491856)
        
        # Assigning a type to the variable 'projected' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'projected', project_to_sphere_call_result_491857)
        
        # Call to assert_array_almost_equal(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'points' (line 52)
        points_491859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 34), 'points', False)
        # Getting the type of 'projected' (line 52)
        projected_491860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 42), 'projected', False)
        # Processing the call keyword arguments (line 52)
        kwargs_491861 = {}
        # Getting the type of 'assert_array_almost_equal' (line 52)
        assert_array_almost_equal_491858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 52)
        assert_array_almost_equal_call_result_491862 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assert_array_almost_equal_491858, *[points_491859, projected_491860], **kwargs_491861)
        
        
        # ################# End of 'test_scaled_points(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scaled_points' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_491863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_491863)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scaled_points'
        return stypy_return_type_491863


    @norecursion
    def test_translated_sphere(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_translated_sphere'
        module_type_store = module_type_store.open_function_context('test_translated_sphere', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_localization', localization)
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_function_name', 'TestProjectToSphere.test_translated_sphere')
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_param_names_list', [])
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProjectToSphere.test_translated_sphere.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProjectToSphere.test_translated_sphere', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_translated_sphere', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_translated_sphere(...)' code ##################

        
        # Assigning a Call to a Name (line 55):
        
        # Call to array(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_491866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_491867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_491868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 27), list_491867, int_491868)
        # Adding element type (line 55)
        int_491869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 27), list_491867, int_491869)
        # Adding element type (line 55)
        int_491870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 27), list_491867, int_491870)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 26), list_491866, list_491867)
        # Adding element type (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_491871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_491872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), list_491871, int_491872)
        # Adding element type (line 55)
        int_491873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), list_491871, int_491873)
        # Adding element type (line 55)
        int_491874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), list_491871, int_491874)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 26), list_491866, list_491871)
        # Adding element type (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_491875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_491876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 49), list_491875, int_491876)
        # Adding element type (line 55)
        int_491877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 49), list_491875, int_491877)
        # Adding element type (line 55)
        int_491878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 49), list_491875, int_491878)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 26), list_491866, list_491875)
        
        # Processing the call keyword arguments (line 55)
        kwargs_491879 = {}
        # Getting the type of 'np' (line 55)
        np_491864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 55)
        array_491865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), np_491864, 'array')
        # Calling array(args, kwargs) (line 55)
        array_call_result_491880 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), array_491865, *[list_491866], **kwargs_491879)
        
        # Assigning a type to the variable 'points' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'points', array_call_result_491880)
        
        # Assigning a Call to a Name (line 56):
        
        # Call to array(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_491883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_491884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 26), list_491883, int_491884)
        # Adding element type (line 56)
        int_491885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 26), list_491883, int_491885)
        # Adding element type (line 56)
        int_491886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 26), list_491883, int_491886)
        
        # Processing the call keyword arguments (line 56)
        kwargs_491887 = {}
        # Getting the type of 'np' (line 56)
        np_491881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 56)
        array_491882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), np_491881, 'array')
        # Calling array(args, kwargs) (line 56)
        array_call_result_491888 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), array_491882, *[list_491883], **kwargs_491887)
        
        # Assigning a type to the variable 'center' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'center', array_call_result_491888)
        
        # Assigning a BinOp to a Name (line 57):
        # Getting the type of 'points' (line 57)
        points_491889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'points')
        # Getting the type of 'center' (line 57)
        center_491890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'center')
        # Applying the binary operator '+' (line 57)
        result_add_491891 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 21), '+', points_491889, center_491890)
        
        # Assigning a type to the variable 'translated' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'translated', result_add_491891)
        
        # Assigning a Num to a Name (line 58):
        int_491892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 17), 'int')
        # Assigning a type to the variable 'radius' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'radius', int_491892)
        
        # Assigning a Call to a Name (line 59):
        
        # Call to project_to_sphere(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'translated' (line 59)
        translated_491895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 56), 'translated', False)
        # Getting the type of 'center' (line 59)
        center_491896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 68), 'center', False)
        # Getting the type of 'radius' (line 60)
        radius_491897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'radius', False)
        # Processing the call keyword arguments (line 59)
        kwargs_491898 = {}
        # Getting the type of 'spherical_voronoi' (line 59)
        spherical_voronoi_491893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'spherical_voronoi', False)
        # Obtaining the member 'project_to_sphere' of a type (line 59)
        project_to_sphere_491894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), spherical_voronoi_491893, 'project_to_sphere')
        # Calling project_to_sphere(args, kwargs) (line 59)
        project_to_sphere_call_result_491899 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), project_to_sphere_491894, *[translated_491895, center_491896, radius_491897], **kwargs_491898)
        
        # Assigning a type to the variable 'projected' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'projected', project_to_sphere_call_result_491899)
        
        # Call to assert_array_almost_equal(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'translated' (line 61)
        translated_491901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'translated', False)
        # Getting the type of 'projected' (line 61)
        projected_491902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'projected', False)
        # Processing the call keyword arguments (line 61)
        kwargs_491903 = {}
        # Getting the type of 'assert_array_almost_equal' (line 61)
        assert_array_almost_equal_491900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 61)
        assert_array_almost_equal_call_result_491904 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_array_almost_equal_491900, *[translated_491901, projected_491902], **kwargs_491903)
        
        
        # ################# End of 'test_translated_sphere(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_translated_sphere' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_491905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_491905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_translated_sphere'
        return stypy_return_type_491905


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 37, 0, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProjectToSphere.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestProjectToSphere' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'TestProjectToSphere', TestProjectToSphere)
# Declaration of the 'TestSphericalVoronoi' class

class TestSphericalVoronoi(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.setup_method')
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 67):
        
        # Call to array(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_491908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_491909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        float_491910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), list_491909, float_491910)
        # Adding element type (line 68)
        float_491911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), list_491909, float_491911)
        # Adding element type (line 68)
        float_491912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), list_491909, float_491912)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_491908, list_491909)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_491913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        float_491914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 12), list_491913, float_491914)
        # Adding element type (line 69)
        float_491915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 12), list_491913, float_491915)
        # Adding element type (line 69)
        float_491916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 12), list_491913, float_491916)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_491908, list_491913)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_491917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        float_491918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), list_491917, float_491918)
        # Adding element type (line 70)
        float_491919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), list_491917, float_491919)
        # Adding element type (line 70)
        float_491920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), list_491917, float_491920)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_491908, list_491917)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_491921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        float_491922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 12), list_491921, float_491922)
        # Adding element type (line 71)
        float_491923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 12), list_491921, float_491923)
        # Adding element type (line 71)
        float_491924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 12), list_491921, float_491924)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_491908, list_491921)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_491925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        float_491926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 12), list_491925, float_491926)
        # Adding element type (line 72)
        float_491927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 12), list_491925, float_491927)
        # Adding element type (line 72)
        float_491928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 12), list_491925, float_491928)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_491908, list_491925)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_491929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        float_491930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), list_491929, float_491930)
        # Adding element type (line 73)
        float_491931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), list_491929, float_491931)
        # Adding element type (line 73)
        float_491932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), list_491929, float_491932)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_491908, list_491929)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_491933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        float_491934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), list_491933, float_491934)
        # Adding element type (line 74)
        float_491935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), list_491933, float_491935)
        # Adding element type (line 74)
        float_491936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), list_491933, float_491936)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_491908, list_491933)
        # Adding element type (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_491937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        float_491938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 12), list_491937, float_491938)
        # Adding element type (line 75)
        float_491939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 12), list_491937, float_491939)
        # Adding element type (line 75)
        float_491940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 12), list_491937, float_491940)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), list_491908, list_491937)
        
        # Processing the call keyword arguments (line 67)
        kwargs_491941 = {}
        # Getting the type of 'np' (line 67)
        np_491906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 67)
        array_491907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 22), np_491906, 'array')
        # Calling array(args, kwargs) (line 67)
        array_call_result_491942 = invoke(stypy.reporting.localization.Localization(__file__, 67, 22), array_491907, *[list_491908], **kwargs_491941)
        
        # Getting the type of 'self' (line 67)
        self_491943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'points' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_491943, 'points', array_call_result_491942)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_491944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_491944)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_491944


    @norecursion
    def test_constructor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_constructor'
        module_type_store = module_type_store.open_function_context('test_constructor', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_constructor')
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_constructor.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_constructor', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_constructor', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_constructor(...)' code ##################

        
        # Assigning a Call to a Name (line 79):
        
        # Call to array(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_491947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        int_491948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 26), list_491947, int_491948)
        # Adding element type (line 79)
        int_491949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 26), list_491947, int_491949)
        # Adding element type (line 79)
        int_491950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 26), list_491947, int_491950)
        
        # Processing the call keyword arguments (line 79)
        kwargs_491951 = {}
        # Getting the type of 'np' (line 79)
        np_491945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 79)
        array_491946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 17), np_491945, 'array')
        # Calling array(args, kwargs) (line 79)
        array_call_result_491952 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), array_491946, *[list_491947], **kwargs_491951)
        
        # Assigning a type to the variable 'center' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'center', array_call_result_491952)
        
        # Assigning a Num to a Name (line 80):
        int_491953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'int')
        # Assigning a type to the variable 'radius' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'radius', int_491953)
        
        # Assigning a Call to a Name (line 81):
        
        # Call to SphericalVoronoi(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'self' (line 81)
        self_491955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'self', False)
        # Obtaining the member 'points' of a type (line 81)
        points_491956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 30), self_491955, 'points')
        # Processing the call keyword arguments (line 81)
        kwargs_491957 = {}
        # Getting the type of 'SphericalVoronoi' (line 81)
        SphericalVoronoi_491954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 81)
        SphericalVoronoi_call_result_491958 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), SphericalVoronoi_491954, *[points_491956], **kwargs_491957)
        
        # Assigning a type to the variable 's1' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 's1', SphericalVoronoi_call_result_491958)
        
        # Assigning a Call to a Name (line 85):
        
        # Call to SphericalVoronoi(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 85)
        self_491960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), 'self', False)
        # Obtaining the member 'points' of a type (line 85)
        points_491961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 30), self_491960, 'points')
        # Getting the type of 'radius' (line 85)
        radius_491962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 44), 'radius', False)
        # Applying the binary operator '*' (line 85)
        result_mul_491963 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 30), '*', points_491961, radius_491962)
        
        # Getting the type of 'radius' (line 85)
        radius_491964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 52), 'radius', False)
        # Processing the call keyword arguments (line 85)
        kwargs_491965 = {}
        # Getting the type of 'SphericalVoronoi' (line 85)
        SphericalVoronoi_491959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 85)
        SphericalVoronoi_call_result_491966 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), SphericalVoronoi_491959, *[result_mul_491963, radius_491964], **kwargs_491965)
        
        # Assigning a type to the variable 's2' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 's2', SphericalVoronoi_call_result_491966)
        
        # Assigning a Call to a Name (line 86):
        
        # Call to SphericalVoronoi(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'self' (line 86)
        self_491968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'self', False)
        # Obtaining the member 'points' of a type (line 86)
        points_491969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 30), self_491968, 'points')
        # Getting the type of 'center' (line 86)
        center_491970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 44), 'center', False)
        # Applying the binary operator '+' (line 86)
        result_add_491971 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 30), '+', points_491969, center_491970)
        
        # Getting the type of 'None' (line 86)
        None_491972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 52), 'None', False)
        # Getting the type of 'center' (line 86)
        center_491973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 58), 'center', False)
        # Processing the call keyword arguments (line 86)
        kwargs_491974 = {}
        # Getting the type of 'SphericalVoronoi' (line 86)
        SphericalVoronoi_491967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 86)
        SphericalVoronoi_call_result_491975 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), SphericalVoronoi_491967, *[result_add_491971, None_491972, center_491973], **kwargs_491974)
        
        # Assigning a type to the variable 's3' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 's3', SphericalVoronoi_call_result_491975)
        
        # Assigning a Call to a Name (line 87):
        
        # Call to SphericalVoronoi(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_491977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'self', False)
        # Obtaining the member 'points' of a type (line 87)
        points_491978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 30), self_491977, 'points')
        # Getting the type of 'radius' (line 87)
        radius_491979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 44), 'radius', False)
        # Applying the binary operator '*' (line 87)
        result_mul_491980 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 30), '*', points_491978, radius_491979)
        
        # Getting the type of 'center' (line 87)
        center_491981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 53), 'center', False)
        # Applying the binary operator '+' (line 87)
        result_add_491982 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 30), '+', result_mul_491980, center_491981)
        
        # Getting the type of 'radius' (line 87)
        radius_491983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 61), 'radius', False)
        # Getting the type of 'center' (line 87)
        center_491984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 69), 'center', False)
        # Processing the call keyword arguments (line 87)
        kwargs_491985 = {}
        # Getting the type of 'SphericalVoronoi' (line 87)
        SphericalVoronoi_491976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 87)
        SphericalVoronoi_call_result_491986 = invoke(stypy.reporting.localization.Localization(__file__, 87, 13), SphericalVoronoi_491976, *[result_add_491982, radius_491983, center_491984], **kwargs_491985)
        
        # Assigning a type to the variable 's4' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 's4', SphericalVoronoi_call_result_491986)
        
        # Call to assert_array_equal(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 's1' (line 88)
        s1_491988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 's1', False)
        # Obtaining the member 'center' of a type (line 88)
        center_491989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 27), s1_491988, 'center')
        
        # Call to array(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_491992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        int_491993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 47), list_491992, int_491993)
        # Adding element type (line 88)
        int_491994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 47), list_491992, int_491994)
        # Adding element type (line 88)
        int_491995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 47), list_491992, int_491995)
        
        # Processing the call keyword arguments (line 88)
        kwargs_491996 = {}
        # Getting the type of 'np' (line 88)
        np_491990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 88)
        array_491991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 38), np_491990, 'array')
        # Calling array(args, kwargs) (line 88)
        array_call_result_491997 = invoke(stypy.reporting.localization.Localization(__file__, 88, 38), array_491991, *[list_491992], **kwargs_491996)
        
        # Processing the call keyword arguments (line 88)
        kwargs_491998 = {}
        # Getting the type of 'assert_array_equal' (line 88)
        assert_array_equal_491987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 88)
        assert_array_equal_call_result_491999 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), assert_array_equal_491987, *[center_491989, array_call_result_491997], **kwargs_491998)
        
        
        # Call to assert_equal(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 's1' (line 89)
        s1_492001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 's1', False)
        # Obtaining the member 'radius' of a type (line 89)
        radius_492002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 21), s1_492001, 'radius')
        int_492003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'int')
        # Processing the call keyword arguments (line 89)
        kwargs_492004 = {}
        # Getting the type of 'assert_equal' (line 89)
        assert_equal_492000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 89)
        assert_equal_call_result_492005 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert_equal_492000, *[radius_492002, int_492003], **kwargs_492004)
        
        
        # Call to assert_array_equal(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 's2' (line 90)
        s2_492007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 's2', False)
        # Obtaining the member 'center' of a type (line 90)
        center_492008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 27), s2_492007, 'center')
        
        # Call to array(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_492011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_492012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 47), list_492011, int_492012)
        # Adding element type (line 90)
        int_492013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 47), list_492011, int_492013)
        # Adding element type (line 90)
        int_492014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 47), list_492011, int_492014)
        
        # Processing the call keyword arguments (line 90)
        kwargs_492015 = {}
        # Getting the type of 'np' (line 90)
        np_492009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 90)
        array_492010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 38), np_492009, 'array')
        # Calling array(args, kwargs) (line 90)
        array_call_result_492016 = invoke(stypy.reporting.localization.Localization(__file__, 90, 38), array_492010, *[list_492011], **kwargs_492015)
        
        # Processing the call keyword arguments (line 90)
        kwargs_492017 = {}
        # Getting the type of 'assert_array_equal' (line 90)
        assert_array_equal_492006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 90)
        assert_array_equal_call_result_492018 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_array_equal_492006, *[center_492008, array_call_result_492016], **kwargs_492017)
        
        
        # Call to assert_equal(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 's2' (line 91)
        s2_492020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 's2', False)
        # Obtaining the member 'radius' of a type (line 91)
        radius_492021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 21), s2_492020, 'radius')
        int_492022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_492023 = {}
        # Getting the type of 'assert_equal' (line 91)
        assert_equal_492019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 91)
        assert_equal_call_result_492024 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), assert_equal_492019, *[radius_492021, int_492022], **kwargs_492023)
        
        
        # Call to assert_array_equal(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 's3' (line 92)
        s3_492026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 's3', False)
        # Obtaining the member 'center' of a type (line 92)
        center_492027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 27), s3_492026, 'center')
        # Getting the type of 'center' (line 92)
        center_492028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 38), 'center', False)
        # Processing the call keyword arguments (line 92)
        kwargs_492029 = {}
        # Getting the type of 'assert_array_equal' (line 92)
        assert_array_equal_492025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 92)
        assert_array_equal_call_result_492030 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), assert_array_equal_492025, *[center_492027, center_492028], **kwargs_492029)
        
        
        # Call to assert_equal(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 's3' (line 93)
        s3_492032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 's3', False)
        # Obtaining the member 'radius' of a type (line 93)
        radius_492033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 21), s3_492032, 'radius')
        int_492034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 32), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_492035 = {}
        # Getting the type of 'assert_equal' (line 93)
        assert_equal_492031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 93)
        assert_equal_call_result_492036 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), assert_equal_492031, *[radius_492033, int_492034], **kwargs_492035)
        
        
        # Call to assert_array_equal(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 's4' (line 94)
        s4_492038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 's4', False)
        # Obtaining the member 'center' of a type (line 94)
        center_492039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 27), s4_492038, 'center')
        # Getting the type of 'center' (line 94)
        center_492040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'center', False)
        # Processing the call keyword arguments (line 94)
        kwargs_492041 = {}
        # Getting the type of 'assert_array_equal' (line 94)
        assert_array_equal_492037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 94)
        assert_array_equal_call_result_492042 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assert_array_equal_492037, *[center_492039, center_492040], **kwargs_492041)
        
        
        # Call to assert_equal(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 's4' (line 95)
        s4_492044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 's4', False)
        # Obtaining the member 'radius' of a type (line 95)
        radius_492045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 21), s4_492044, 'radius')
        # Getting the type of 'radius' (line 95)
        radius_492046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'radius', False)
        # Processing the call keyword arguments (line 95)
        kwargs_492047 = {}
        # Getting the type of 'assert_equal' (line 95)
        assert_equal_492043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 95)
        assert_equal_call_result_492048 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assert_equal_492043, *[radius_492045, radius_492046], **kwargs_492047)
        
        
        # ################# End of 'test_constructor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_constructor' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_492049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492049)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_constructor'
        return stypy_return_type_492049


    @norecursion
    def test_vertices_regions_translation_invariance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vertices_regions_translation_invariance'
        module_type_store = module_type_store.open_function_context('test_vertices_regions_translation_invariance', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_vertices_regions_translation_invariance')
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_vertices_regions_translation_invariance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_vertices_regions_translation_invariance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vertices_regions_translation_invariance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vertices_regions_translation_invariance(...)' code ##################

        
        # Assigning a Call to a Name (line 98):
        
        # Call to SphericalVoronoi(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'self' (line 98)
        self_492051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'self', False)
        # Obtaining the member 'points' of a type (line 98)
        points_492052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 37), self_492051, 'points')
        # Processing the call keyword arguments (line 98)
        kwargs_492053 = {}
        # Getting the type of 'SphericalVoronoi' (line 98)
        SphericalVoronoi_492050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 98)
        SphericalVoronoi_call_result_492054 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), SphericalVoronoi_492050, *[points_492052], **kwargs_492053)
        
        # Assigning a type to the variable 'sv_origin' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'sv_origin', SphericalVoronoi_call_result_492054)
        
        # Assigning a Call to a Name (line 99):
        
        # Call to array(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_492057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        int_492058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 26), list_492057, int_492058)
        # Adding element type (line 99)
        int_492059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 26), list_492057, int_492059)
        # Adding element type (line 99)
        int_492060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 26), list_492057, int_492060)
        
        # Processing the call keyword arguments (line 99)
        kwargs_492061 = {}
        # Getting the type of 'np' (line 99)
        np_492055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 99)
        array_492056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), np_492055, 'array')
        # Calling array(args, kwargs) (line 99)
        array_call_result_492062 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), array_492056, *[list_492057], **kwargs_492061)
        
        # Assigning a type to the variable 'center' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'center', array_call_result_492062)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to SphericalVoronoi(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'self' (line 100)
        self_492064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 41), 'self', False)
        # Obtaining the member 'points' of a type (line 100)
        points_492065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 41), self_492064, 'points')
        # Getting the type of 'center' (line 100)
        center_492066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 55), 'center', False)
        # Applying the binary operator '+' (line 100)
        result_add_492067 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 41), '+', points_492065, center_492066)
        
        # Getting the type of 'None' (line 100)
        None_492068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 63), 'None', False)
        # Getting the type of 'center' (line 100)
        center_492069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 69), 'center', False)
        # Processing the call keyword arguments (line 100)
        kwargs_492070 = {}
        # Getting the type of 'SphericalVoronoi' (line 100)
        SphericalVoronoi_492063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 100)
        SphericalVoronoi_call_result_492071 = invoke(stypy.reporting.localization.Localization(__file__, 100, 24), SphericalVoronoi_492063, *[result_add_492067, None_492068, center_492069], **kwargs_492070)
        
        # Assigning a type to the variable 'sv_translated' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'sv_translated', SphericalVoronoi_call_result_492071)
        
        # Call to assert_array_equal(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'sv_origin' (line 101)
        sv_origin_492073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'sv_origin', False)
        # Obtaining the member 'regions' of a type (line 101)
        regions_492074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), sv_origin_492073, 'regions')
        # Getting the type of 'sv_translated' (line 101)
        sv_translated_492075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'sv_translated', False)
        # Obtaining the member 'regions' of a type (line 101)
        regions_492076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 46), sv_translated_492075, 'regions')
        # Processing the call keyword arguments (line 101)
        kwargs_492077 = {}
        # Getting the type of 'assert_array_equal' (line 101)
        assert_array_equal_492072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 101)
        assert_array_equal_call_result_492078 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assert_array_equal_492072, *[regions_492074, regions_492076], **kwargs_492077)
        
        
        # Call to assert_array_almost_equal(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'sv_origin' (line 102)
        sv_origin_492080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'sv_origin', False)
        # Obtaining the member 'vertices' of a type (line 102)
        vertices_492081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 34), sv_origin_492080, 'vertices')
        # Getting the type of 'center' (line 102)
        center_492082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 55), 'center', False)
        # Applying the binary operator '+' (line 102)
        result_add_492083 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 34), '+', vertices_492081, center_492082)
        
        # Getting the type of 'sv_translated' (line 103)
        sv_translated_492084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'sv_translated', False)
        # Obtaining the member 'vertices' of a type (line 103)
        vertices_492085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 34), sv_translated_492084, 'vertices')
        # Processing the call keyword arguments (line 102)
        kwargs_492086 = {}
        # Getting the type of 'assert_array_almost_equal' (line 102)
        assert_array_almost_equal_492079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 102)
        assert_array_almost_equal_call_result_492087 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_array_almost_equal_492079, *[result_add_492083, vertices_492085], **kwargs_492086)
        
        
        # ################# End of 'test_vertices_regions_translation_invariance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vertices_regions_translation_invariance' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_492088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vertices_regions_translation_invariance'
        return stypy_return_type_492088


    @norecursion
    def test_vertices_regions_scaling_invariance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vertices_regions_scaling_invariance'
        module_type_store = module_type_store.open_function_context('test_vertices_regions_scaling_invariance', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_vertices_regions_scaling_invariance')
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_vertices_regions_scaling_invariance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_vertices_regions_scaling_invariance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vertices_regions_scaling_invariance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vertices_regions_scaling_invariance(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Call to SphericalVoronoi(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'self' (line 106)
        self_492090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'self', False)
        # Obtaining the member 'points' of a type (line 106)
        points_492091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 35), self_492090, 'points')
        # Processing the call keyword arguments (line 106)
        kwargs_492092 = {}
        # Getting the type of 'SphericalVoronoi' (line 106)
        SphericalVoronoi_492089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 106)
        SphericalVoronoi_call_result_492093 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), SphericalVoronoi_492089, *[points_492091], **kwargs_492092)
        
        # Assigning a type to the variable 'sv_unit' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'sv_unit', SphericalVoronoi_call_result_492093)
        
        # Assigning a Call to a Name (line 107):
        
        # Call to SphericalVoronoi(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'self' (line 107)
        self_492095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'self', False)
        # Obtaining the member 'points' of a type (line 107)
        points_492096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 37), self_492095, 'points')
        int_492097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 51), 'int')
        # Applying the binary operator '*' (line 107)
        result_mul_492098 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 37), '*', points_492096, int_492097)
        
        int_492099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 54), 'int')
        # Processing the call keyword arguments (line 107)
        kwargs_492100 = {}
        # Getting the type of 'SphericalVoronoi' (line 107)
        SphericalVoronoi_492094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 20), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 107)
        SphericalVoronoi_call_result_492101 = invoke(stypy.reporting.localization.Localization(__file__, 107, 20), SphericalVoronoi_492094, *[result_mul_492098, int_492099], **kwargs_492100)
        
        # Assigning a type to the variable 'sv_scaled' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'sv_scaled', SphericalVoronoi_call_result_492101)
        
        # Call to assert_array_equal(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'sv_unit' (line 108)
        sv_unit_492103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'sv_unit', False)
        # Obtaining the member 'regions' of a type (line 108)
        regions_492104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), sv_unit_492103, 'regions')
        # Getting the type of 'sv_scaled' (line 108)
        sv_scaled_492105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'sv_scaled', False)
        # Obtaining the member 'regions' of a type (line 108)
        regions_492106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 44), sv_scaled_492105, 'regions')
        # Processing the call keyword arguments (line 108)
        kwargs_492107 = {}
        # Getting the type of 'assert_array_equal' (line 108)
        assert_array_equal_492102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 108)
        assert_array_equal_call_result_492108 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assert_array_equal_492102, *[regions_492104, regions_492106], **kwargs_492107)
        
        
        # Call to assert_array_almost_equal(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'sv_unit' (line 109)
        sv_unit_492110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'sv_unit', False)
        # Obtaining the member 'vertices' of a type (line 109)
        vertices_492111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 34), sv_unit_492110, 'vertices')
        int_492112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 53), 'int')
        # Applying the binary operator '*' (line 109)
        result_mul_492113 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 34), '*', vertices_492111, int_492112)
        
        # Getting the type of 'sv_scaled' (line 110)
        sv_scaled_492114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'sv_scaled', False)
        # Obtaining the member 'vertices' of a type (line 110)
        vertices_492115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 34), sv_scaled_492114, 'vertices')
        # Processing the call keyword arguments (line 109)
        kwargs_492116 = {}
        # Getting the type of 'assert_array_almost_equal' (line 109)
        assert_array_almost_equal_492109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 109)
        assert_array_almost_equal_call_result_492117 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), assert_array_almost_equal_492109, *[result_mul_492113, vertices_492115], **kwargs_492116)
        
        
        # ################# End of 'test_vertices_regions_scaling_invariance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vertices_regions_scaling_invariance' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_492118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492118)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vertices_regions_scaling_invariance'
        return stypy_return_type_492118


    @norecursion
    def test_sort_vertices_of_regions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sort_vertices_of_regions'
        module_type_store = module_type_store.open_function_context('test_sort_vertices_of_regions', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_sort_vertices_of_regions')
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_sort_vertices_of_regions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_sort_vertices_of_regions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sort_vertices_of_regions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sort_vertices_of_regions(...)' code ##################

        
        # Assigning a Call to a Name (line 113):
        
        # Call to SphericalVoronoi(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_492120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'self', False)
        # Obtaining the member 'points' of a type (line 113)
        points_492121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 30), self_492120, 'points')
        # Processing the call keyword arguments (line 113)
        kwargs_492122 = {}
        # Getting the type of 'SphericalVoronoi' (line 113)
        SphericalVoronoi_492119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 113)
        SphericalVoronoi_call_result_492123 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), SphericalVoronoi_492119, *[points_492121], **kwargs_492122)
        
        # Assigning a type to the variable 'sv' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'sv', SphericalVoronoi_call_result_492123)
        
        # Assigning a Attribute to a Name (line 114):
        # Getting the type of 'sv' (line 114)
        sv_492124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), 'sv')
        # Obtaining the member 'regions' of a type (line 114)
        regions_492125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 27), sv_492124, 'regions')
        # Assigning a type to the variable 'unsorted_regions' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'unsorted_regions', regions_492125)
        
        # Call to sort_vertices_of_regions(...): (line 115)
        # Processing the call keyword arguments (line 115)
        kwargs_492128 = {}
        # Getting the type of 'sv' (line 115)
        sv_492126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'sv', False)
        # Obtaining the member 'sort_vertices_of_regions' of a type (line 115)
        sort_vertices_of_regions_492127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), sv_492126, 'sort_vertices_of_regions')
        # Calling sort_vertices_of_regions(args, kwargs) (line 115)
        sort_vertices_of_regions_call_result_492129 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), sort_vertices_of_regions_492127, *[], **kwargs_492128)
        
        
        # Call to assert_array_equal(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to sorted(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'sv' (line 116)
        sv_492132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'sv', False)
        # Obtaining the member 'regions' of a type (line 116)
        regions_492133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), sv_492132, 'regions')
        # Processing the call keyword arguments (line 116)
        kwargs_492134 = {}
        # Getting the type of 'sorted' (line 116)
        sorted_492131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 27), 'sorted', False)
        # Calling sorted(args, kwargs) (line 116)
        sorted_call_result_492135 = invoke(stypy.reporting.localization.Localization(__file__, 116, 27), sorted_492131, *[regions_492133], **kwargs_492134)
        
        
        # Call to sorted(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'unsorted_regions' (line 116)
        unsorted_regions_492137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 54), 'unsorted_regions', False)
        # Processing the call keyword arguments (line 116)
        kwargs_492138 = {}
        # Getting the type of 'sorted' (line 116)
        sorted_492136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'sorted', False)
        # Calling sorted(args, kwargs) (line 116)
        sorted_call_result_492139 = invoke(stypy.reporting.localization.Localization(__file__, 116, 47), sorted_492136, *[unsorted_regions_492137], **kwargs_492138)
        
        # Processing the call keyword arguments (line 116)
        kwargs_492140 = {}
        # Getting the type of 'assert_array_equal' (line 116)
        assert_array_equal_492130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 116)
        assert_array_equal_call_result_492141 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert_array_equal_492130, *[sorted_call_result_492135, sorted_call_result_492139], **kwargs_492140)
        
        
        # ################# End of 'test_sort_vertices_of_regions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sort_vertices_of_regions' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_492142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492142)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sort_vertices_of_regions'
        return stypy_return_type_492142


    @norecursion
    def test_sort_vertices_of_regions_flattened(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sort_vertices_of_regions_flattened'
        module_type_store = module_type_store.open_function_context('test_sort_vertices_of_regions_flattened', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_sort_vertices_of_regions_flattened')
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_sort_vertices_of_regions_flattened.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_sort_vertices_of_regions_flattened', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sort_vertices_of_regions_flattened', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sort_vertices_of_regions_flattened(...)' code ##################

        
        # Assigning a Call to a Name (line 119):
        
        # Call to sorted(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_492144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_492145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_492146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 27), list_492145, int_492146)
        # Adding element type (line 119)
        int_492147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 27), list_492145, int_492147)
        # Adding element type (line 119)
        int_492148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 27), list_492145, int_492148)
        # Adding element type (line 119)
        int_492149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 27), list_492145, int_492149)
        # Adding element type (line 119)
        int_492150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 27), list_492145, int_492150)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_492144, list_492145)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_492151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_492152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), list_492151, int_492152)
        # Adding element type (line 119)
        int_492153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), list_492151, int_492153)
        # Adding element type (line 119)
        int_492154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), list_492151, int_492154)
        # Adding element type (line 119)
        int_492155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), list_492151, int_492155)
        # Adding element type (line 119)
        int_492156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), list_492151, int_492156)
        # Adding element type (line 119)
        int_492157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), list_492151, int_492157)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_492144, list_492151)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_492158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_492159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 66), list_492158, int_492159)
        # Adding element type (line 119)
        int_492160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 66), list_492158, int_492160)
        # Adding element type (line 119)
        int_492161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 73), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 66), list_492158, int_492161)
        # Adding element type (line 119)
        int_492162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 76), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 66), list_492158, int_492162)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_492144, list_492158)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_492163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 80), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_492164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 81), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 80), list_492163, int_492164)
        # Adding element type (line 119)
        int_492165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 84), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 80), list_492163, int_492165)
        # Adding element type (line 119)
        int_492166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 80), list_492163, int_492166)
        # Adding element type (line 119)
        int_492167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 80), list_492163, int_492167)
        # Adding element type (line 119)
        int_492168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 80), list_492163, int_492168)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_492144, list_492163)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_492169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_492170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 22), list_492169, int_492170)
        # Adding element type (line 120)
        int_492171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 22), list_492169, int_492171)
        # Adding element type (line 120)
        int_492172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 22), list_492169, int_492172)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_492144, list_492169)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_492173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_492174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 35), list_492173, int_492174)
        # Adding element type (line 120)
        int_492175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 35), list_492173, int_492175)
        # Adding element type (line 120)
        int_492176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 35), list_492173, int_492176)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_492144, list_492173)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_492177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_492178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 46), list_492177, int_492178)
        # Adding element type (line 120)
        int_492179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 46), list_492177, int_492179)
        # Adding element type (line 120)
        int_492180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 46), list_492177, int_492180)
        # Adding element type (line 120)
        int_492181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 46), list_492177, int_492181)
        # Adding element type (line 120)
        int_492182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 46), list_492177, int_492182)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_492144, list_492177)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_492183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_492184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 64), list_492183, int_492184)
        # Adding element type (line 120)
        int_492185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 64), list_492183, int_492185)
        # Adding element type (line 120)
        int_492186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 71), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 64), list_492183, int_492186)
        # Adding element type (line 120)
        int_492187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 75), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 64), list_492183, int_492187)
        # Adding element type (line 120)
        int_492188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 64), list_492183, int_492188)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_492144, list_492183)
        
        # Processing the call keyword arguments (line 119)
        kwargs_492189 = {}
        # Getting the type of 'sorted' (line 119)
        sorted_492143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 119)
        sorted_call_result_492190 = invoke(stypy.reporting.localization.Localization(__file__, 119, 19), sorted_492143, *[list_492144], **kwargs_492189)
        
        # Assigning a type to the variable 'expected' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'expected', sorted_call_result_492190)
        
        # Assigning a Call to a Name (line 122):
        
        # Call to list(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to chain(...): (line 122)
        
        # Call to sorted(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'expected' (line 122)
        expected_492195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 48), 'expected', False)
        # Processing the call keyword arguments (line 122)
        kwargs_492196 = {}
        # Getting the type of 'sorted' (line 122)
        sorted_492194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 41), 'sorted', False)
        # Calling sorted(args, kwargs) (line 122)
        sorted_call_result_492197 = invoke(stypy.reporting.localization.Localization(__file__, 122, 41), sorted_492194, *[expected_492195], **kwargs_492196)
        
        # Processing the call keyword arguments (line 122)
        kwargs_492198 = {}
        # Getting the type of 'itertools' (line 122)
        itertools_492192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'itertools', False)
        # Obtaining the member 'chain' of a type (line 122)
        chain_492193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 24), itertools_492192, 'chain')
        # Calling chain(args, kwargs) (line 122)
        chain_call_result_492199 = invoke(stypy.reporting.localization.Localization(__file__, 122, 24), chain_492193, *[sorted_call_result_492197], **kwargs_492198)
        
        # Processing the call keyword arguments (line 122)
        kwargs_492200 = {}
        # Getting the type of 'list' (line 122)
        list_492191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'list', False)
        # Calling list(args, kwargs) (line 122)
        list_call_result_492201 = invoke(stypy.reporting.localization.Localization(__file__, 122, 19), list_492191, *[chain_call_result_492199], **kwargs_492200)
        
        # Assigning a type to the variable 'expected' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'expected', list_call_result_492201)
        
        # Assigning a Call to a Name (line 123):
        
        # Call to SphericalVoronoi(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'self' (line 123)
        self_492203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'self', False)
        # Obtaining the member 'points' of a type (line 123)
        points_492204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 30), self_492203, 'points')
        # Processing the call keyword arguments (line 123)
        kwargs_492205 = {}
        # Getting the type of 'SphericalVoronoi' (line 123)
        SphericalVoronoi_492202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 13), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 123)
        SphericalVoronoi_call_result_492206 = invoke(stypy.reporting.localization.Localization(__file__, 123, 13), SphericalVoronoi_492202, *[points_492204], **kwargs_492205)
        
        # Assigning a type to the variable 'sv' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'sv', SphericalVoronoi_call_result_492206)
        
        # Call to sort_vertices_of_regions(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_492209 = {}
        # Getting the type of 'sv' (line 124)
        sv_492207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'sv', False)
        # Obtaining the member 'sort_vertices_of_regions' of a type (line 124)
        sort_vertices_of_regions_492208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), sv_492207, 'sort_vertices_of_regions')
        # Calling sort_vertices_of_regions(args, kwargs) (line 124)
        sort_vertices_of_regions_call_result_492210 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), sort_vertices_of_regions_492208, *[], **kwargs_492209)
        
        
        # Assigning a Call to a Name (line 125):
        
        # Call to list(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to chain(...): (line 125)
        
        # Call to sorted(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'sv' (line 125)
        sv_492215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 46), 'sv', False)
        # Obtaining the member 'regions' of a type (line 125)
        regions_492216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 46), sv_492215, 'regions')
        # Processing the call keyword arguments (line 125)
        kwargs_492217 = {}
        # Getting the type of 'sorted' (line 125)
        sorted_492214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 39), 'sorted', False)
        # Calling sorted(args, kwargs) (line 125)
        sorted_call_result_492218 = invoke(stypy.reporting.localization.Localization(__file__, 125, 39), sorted_492214, *[regions_492216], **kwargs_492217)
        
        # Processing the call keyword arguments (line 125)
        kwargs_492219 = {}
        # Getting the type of 'itertools' (line 125)
        itertools_492212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'itertools', False)
        # Obtaining the member 'chain' of a type (line 125)
        chain_492213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 22), itertools_492212, 'chain')
        # Calling chain(args, kwargs) (line 125)
        chain_call_result_492220 = invoke(stypy.reporting.localization.Localization(__file__, 125, 22), chain_492213, *[sorted_call_result_492218], **kwargs_492219)
        
        # Processing the call keyword arguments (line 125)
        kwargs_492221 = {}
        # Getting the type of 'list' (line 125)
        list_492211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'list', False)
        # Calling list(args, kwargs) (line 125)
        list_call_result_492222 = invoke(stypy.reporting.localization.Localization(__file__, 125, 17), list_492211, *[chain_call_result_492220], **kwargs_492221)
        
        # Assigning a type to the variable 'actual' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'actual', list_call_result_492222)
        
        # Call to assert_array_equal(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'actual' (line 126)
        actual_492224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'actual', False)
        # Getting the type of 'expected' (line 126)
        expected_492225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 35), 'expected', False)
        # Processing the call keyword arguments (line 126)
        kwargs_492226 = {}
        # Getting the type of 'assert_array_equal' (line 126)
        assert_array_equal_492223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 126)
        assert_array_equal_call_result_492227 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assert_array_equal_492223, *[actual_492224, expected_492225], **kwargs_492226)
        
        
        # ################# End of 'test_sort_vertices_of_regions_flattened(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sort_vertices_of_regions_flattened' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_492228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sort_vertices_of_regions_flattened'
        return stypy_return_type_492228


    @norecursion
    def test_num_vertices(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_num_vertices'
        module_type_store = module_type_store.open_function_context('test_num_vertices', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_num_vertices')
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_num_vertices.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_num_vertices', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_num_vertices', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_num_vertices(...)' code ##################

        
        # Assigning a Call to a Name (line 134):
        
        # Call to SphericalVoronoi(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'self' (line 134)
        self_492230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 30), 'self', False)
        # Obtaining the member 'points' of a type (line 134)
        points_492231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 30), self_492230, 'points')
        # Processing the call keyword arguments (line 134)
        kwargs_492232 = {}
        # Getting the type of 'SphericalVoronoi' (line 134)
        SphericalVoronoi_492229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'SphericalVoronoi', False)
        # Calling SphericalVoronoi(args, kwargs) (line 134)
        SphericalVoronoi_call_result_492233 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), SphericalVoronoi_492229, *[points_492231], **kwargs_492232)
        
        # Assigning a type to the variable 'sv' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'sv', SphericalVoronoi_call_result_492233)
        
        # Assigning a BinOp to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_492234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 37), 'int')
        # Getting the type of 'self' (line 135)
        self_492235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'self')
        # Obtaining the member 'points' of a type (line 135)
        points_492236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 19), self_492235, 'points')
        # Obtaining the member 'shape' of a type (line 135)
        shape_492237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 19), points_492236, 'shape')
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___492238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 19), shape_492237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_492239 = invoke(stypy.reporting.localization.Localization(__file__, 135, 19), getitem___492238, int_492234)
        
        int_492240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 42), 'int')
        # Applying the binary operator '*' (line 135)
        result_mul_492241 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 19), '*', subscript_call_result_492239, int_492240)
        
        int_492242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 46), 'int')
        # Applying the binary operator '-' (line 135)
        result_sub_492243 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 19), '-', result_mul_492241, int_492242)
        
        # Assigning a type to the variable 'expected' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'expected', result_sub_492243)
        
        # Assigning a Subscript to a Name (line 136):
        
        # Obtaining the type of the subscript
        int_492244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 35), 'int')
        # Getting the type of 'sv' (line 136)
        sv_492245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'sv')
        # Obtaining the member 'vertices' of a type (line 136)
        vertices_492246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 17), sv_492245, 'vertices')
        # Obtaining the member 'shape' of a type (line 136)
        shape_492247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 17), vertices_492246, 'shape')
        # Obtaining the member '__getitem__' of a type (line 136)
        getitem___492248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 17), shape_492247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 136)
        subscript_call_result_492249 = invoke(stypy.reporting.localization.Localization(__file__, 136, 17), getitem___492248, int_492244)
        
        # Assigning a type to the variable 'actual' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'actual', subscript_call_result_492249)
        
        # Call to assert_equal(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'actual' (line 137)
        actual_492251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'actual', False)
        # Getting the type of 'expected' (line 137)
        expected_492252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'expected', False)
        # Processing the call keyword arguments (line 137)
        kwargs_492253 = {}
        # Getting the type of 'assert_equal' (line 137)
        assert_equal_492250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 137)
        assert_equal_call_result_492254 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert_equal_492250, *[actual_492251, expected_492252], **kwargs_492253)
        
        
        # ################# End of 'test_num_vertices(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_num_vertices' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_492255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492255)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_num_vertices'
        return stypy_return_type_492255


    @norecursion
    def test_voronoi_circles(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_voronoi_circles'
        module_type_store = module_type_store.open_function_context('test_voronoi_circles', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_voronoi_circles')
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_voronoi_circles.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_voronoi_circles', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_voronoi_circles', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_voronoi_circles(...)' code ##################

        
        # Assigning a Call to a Name (line 140):
        
        # Call to SphericalVoronoi(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_492258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 48), 'self', False)
        # Obtaining the member 'points' of a type (line 140)
        points_492259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 48), self_492258, 'points')
        # Processing the call keyword arguments (line 140)
        kwargs_492260 = {}
        # Getting the type of 'spherical_voronoi' (line 140)
        spherical_voronoi_492256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 13), 'spherical_voronoi', False)
        # Obtaining the member 'SphericalVoronoi' of a type (line 140)
        SphericalVoronoi_492257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 13), spherical_voronoi_492256, 'SphericalVoronoi')
        # Calling SphericalVoronoi(args, kwargs) (line 140)
        SphericalVoronoi_call_result_492261 = invoke(stypy.reporting.localization.Localization(__file__, 140, 13), SphericalVoronoi_492257, *[points_492259], **kwargs_492260)
        
        # Assigning a type to the variable 'sv' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'sv', SphericalVoronoi_call_result_492261)
        
        # Getting the type of 'sv' (line 141)
        sv_492262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'sv')
        # Obtaining the member 'vertices' of a type (line 141)
        vertices_492263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 22), sv_492262, 'vertices')
        # Testing the type of a for loop iterable (line 141)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 8), vertices_492263)
        # Getting the type of the for loop variable (line 141)
        for_loop_var_492264 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 8), vertices_492263)
        # Assigning a type to the variable 'vertex' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'vertex', for_loop_var_492264)
        # SSA begins for a for statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 142):
        
        # Call to cdist(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'sv' (line 142)
        sv_492267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'sv', False)
        # Obtaining the member 'points' of a type (line 142)
        points_492268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 39), sv_492267, 'points')
        
        # Call to array(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_492271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of 'vertex' (line 142)
        vertex_492272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 59), 'vertex', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 58), list_492271, vertex_492272)
        
        # Processing the call keyword arguments (line 142)
        kwargs_492273 = {}
        # Getting the type of 'np' (line 142)
        np_492269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 49), 'np', False)
        # Obtaining the member 'array' of a type (line 142)
        array_492270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 49), np_492269, 'array')
        # Calling array(args, kwargs) (line 142)
        array_call_result_492274 = invoke(stypy.reporting.localization.Localization(__file__, 142, 49), array_492270, *[list_492271], **kwargs_492273)
        
        # Processing the call keyword arguments (line 142)
        kwargs_492275 = {}
        # Getting the type of 'distance' (line 142)
        distance_492265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'distance', False)
        # Obtaining the member 'cdist' of a type (line 142)
        cdist_492266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 24), distance_492265, 'cdist')
        # Calling cdist(args, kwargs) (line 142)
        cdist_call_result_492276 = invoke(stypy.reporting.localization.Localization(__file__, 142, 24), cdist_492266, *[points_492268, array_call_result_492274], **kwargs_492275)
        
        # Assigning a type to the variable 'distances' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'distances', cdist_call_result_492276)
        
        # Assigning a Call to a Name (line 143):
        
        # Call to array(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining the type of the subscript
        int_492279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 49), 'int')
        int_492280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 51), 'int')
        slice_492281 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 143, 31), int_492279, int_492280, None)
        
        # Call to sorted(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'distances' (line 143)
        distances_492283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 38), 'distances', False)
        # Processing the call keyword arguments (line 143)
        kwargs_492284 = {}
        # Getting the type of 'sorted' (line 143)
        sorted_492282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'sorted', False)
        # Calling sorted(args, kwargs) (line 143)
        sorted_call_result_492285 = invoke(stypy.reporting.localization.Localization(__file__, 143, 31), sorted_492282, *[distances_492283], **kwargs_492284)
        
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___492286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 31), sorted_call_result_492285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_492287 = invoke(stypy.reporting.localization.Localization(__file__, 143, 31), getitem___492286, slice_492281)
        
        # Processing the call keyword arguments (line 143)
        kwargs_492288 = {}
        # Getting the type of 'np' (line 143)
        np_492277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 143)
        array_492278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 22), np_492277, 'array')
        # Calling array(args, kwargs) (line 143)
        array_call_result_492289 = invoke(stypy.reporting.localization.Localization(__file__, 143, 22), array_492278, *[subscript_call_result_492287], **kwargs_492288)
        
        # Assigning a type to the variable 'closest' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'closest', array_call_result_492289)
        
        # Call to assert_almost_equal(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining the type of the subscript
        int_492291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 40), 'int')
        # Getting the type of 'closest' (line 144)
        closest_492292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'closest', False)
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___492293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 32), closest_492292, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_492294 = invoke(stypy.reporting.localization.Localization(__file__, 144, 32), getitem___492293, int_492291)
        
        
        # Obtaining the type of the subscript
        int_492295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 52), 'int')
        # Getting the type of 'closest' (line 144)
        closest_492296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 44), 'closest', False)
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___492297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 44), closest_492296, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_492298 = invoke(stypy.reporting.localization.Localization(__file__, 144, 44), getitem___492297, int_492295)
        
        int_492299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 56), 'int')
        
        # Call to str(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'vertex' (line 144)
        vertex_492301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 63), 'vertex', False)
        # Processing the call keyword arguments (line 144)
        kwargs_492302 = {}
        # Getting the type of 'str' (line 144)
        str_492300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 59), 'str', False)
        # Calling str(args, kwargs) (line 144)
        str_call_result_492303 = invoke(stypy.reporting.localization.Localization(__file__, 144, 59), str_492300, *[vertex_492301], **kwargs_492302)
        
        # Processing the call keyword arguments (line 144)
        kwargs_492304 = {}
        # Getting the type of 'assert_almost_equal' (line 144)
        assert_almost_equal_492290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 144)
        assert_almost_equal_call_result_492305 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), assert_almost_equal_492290, *[subscript_call_result_492294, subscript_call_result_492298, int_492299, str_call_result_492303], **kwargs_492304)
        
        
        # Call to assert_almost_equal(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Obtaining the type of the subscript
        int_492307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 40), 'int')
        # Getting the type of 'closest' (line 145)
        closest_492308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'closest', False)
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___492309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 32), closest_492308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_492310 = invoke(stypy.reporting.localization.Localization(__file__, 145, 32), getitem___492309, int_492307)
        
        
        # Obtaining the type of the subscript
        int_492311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 52), 'int')
        # Getting the type of 'closest' (line 145)
        closest_492312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 44), 'closest', False)
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___492313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 44), closest_492312, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_492314 = invoke(stypy.reporting.localization.Localization(__file__, 145, 44), getitem___492313, int_492311)
        
        int_492315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 56), 'int')
        
        # Call to str(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'vertex' (line 145)
        vertex_492317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 63), 'vertex', False)
        # Processing the call keyword arguments (line 145)
        kwargs_492318 = {}
        # Getting the type of 'str' (line 145)
        str_492316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 59), 'str', False)
        # Calling str(args, kwargs) (line 145)
        str_call_result_492319 = invoke(stypy.reporting.localization.Localization(__file__, 145, 59), str_492316, *[vertex_492317], **kwargs_492318)
        
        # Processing the call keyword arguments (line 145)
        kwargs_492320 = {}
        # Getting the type of 'assert_almost_equal' (line 145)
        assert_almost_equal_492306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 145)
        assert_almost_equal_call_result_492321 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), assert_almost_equal_492306, *[subscript_call_result_492310, subscript_call_result_492314, int_492315, str_call_result_492319], **kwargs_492320)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_voronoi_circles(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_voronoi_circles' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_492322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_voronoi_circles'
        return stypy_return_type_492322


    @norecursion
    def test_duplicate_point_handling(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_duplicate_point_handling'
        module_type_store = module_type_store.open_function_context('test_duplicate_point_handling', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_duplicate_point_handling')
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_duplicate_point_handling.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_duplicate_point_handling', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_duplicate_point_handling', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_duplicate_point_handling(...)' code ##################

        
        # Assigning a Call to a Attribute (line 150):
        
        # Call to concatenate(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_492325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        # Getting the type of 'self' (line 150)
        self_492326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 42), 'self', False)
        # Obtaining the member 'points' of a type (line 150)
        points_492327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 42), self_492326, 'points')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 42), tuple_492325, points_492327)
        # Adding element type (line 150)
        # Getting the type of 'self' (line 150)
        self_492328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 55), 'self', False)
        # Obtaining the member 'points' of a type (line 150)
        points_492329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 55), self_492328, 'points')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 42), tuple_492325, points_492329)
        
        # Processing the call keyword arguments (line 150)
        kwargs_492330 = {}
        # Getting the type of 'np' (line 150)
        np_492323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 150)
        concatenate_492324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 26), np_492323, 'concatenate')
        # Calling concatenate(args, kwargs) (line 150)
        concatenate_call_result_492331 = invoke(stypy.reporting.localization.Localization(__file__, 150, 26), concatenate_492324, *[tuple_492325], **kwargs_492330)
        
        # Getting the type of 'self' (line 150)
        self_492332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'degenerate' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_492332, 'degenerate', concatenate_call_result_492331)
        
        # Call to assert_raises(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'ValueError' (line 151)
        ValueError_492334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 151)
        kwargs_492335 = {}
        # Getting the type of 'assert_raises' (line 151)
        assert_raises_492333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 151)
        assert_raises_call_result_492336 = invoke(stypy.reporting.localization.Localization(__file__, 151, 13), assert_raises_492333, *[ValueError_492334], **kwargs_492335)
        
        with_492337 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 151, 13), assert_raises_call_result_492336, 'with parameter', '__enter__', '__exit__')

        if with_492337:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 151)
            enter___492338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 13), assert_raises_call_result_492336, '__enter__')
            with_enter_492339 = invoke(stypy.reporting.localization.Localization(__file__, 151, 13), enter___492338)
            
            # Assigning a Call to a Name (line 152):
            
            # Call to SphericalVoronoi(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'self' (line 152)
            self_492342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 52), 'self', False)
            # Obtaining the member 'degenerate' of a type (line 152)
            degenerate_492343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 52), self_492342, 'degenerate')
            # Processing the call keyword arguments (line 152)
            kwargs_492344 = {}
            # Getting the type of 'spherical_voronoi' (line 152)
            spherical_voronoi_492340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'spherical_voronoi', False)
            # Obtaining the member 'SphericalVoronoi' of a type (line 152)
            SphericalVoronoi_492341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 17), spherical_voronoi_492340, 'SphericalVoronoi')
            # Calling SphericalVoronoi(args, kwargs) (line 152)
            SphericalVoronoi_call_result_492345 = invoke(stypy.reporting.localization.Localization(__file__, 152, 17), SphericalVoronoi_492341, *[degenerate_492343], **kwargs_492344)
            
            # Assigning a type to the variable 'sv' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'sv', SphericalVoronoi_call_result_492345)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 151)
            exit___492346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 13), assert_raises_call_result_492336, '__exit__')
            with_exit_492347 = invoke(stypy.reporting.localization.Localization(__file__, 151, 13), exit___492346, None, None, None)

        
        # ################# End of 'test_duplicate_point_handling(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_duplicate_point_handling' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_492348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492348)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_duplicate_point_handling'
        return stypy_return_type_492348


    @norecursion
    def test_incorrect_radius_handling(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_incorrect_radius_handling'
        module_type_store = module_type_store.open_function_context('test_incorrect_radius_handling', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_incorrect_radius_handling')
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_incorrect_radius_handling.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_incorrect_radius_handling', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_incorrect_radius_handling', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_incorrect_radius_handling(...)' code ##################

        
        # Call to assert_raises(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'ValueError' (line 157)
        ValueError_492350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 157)
        kwargs_492351 = {}
        # Getting the type of 'assert_raises' (line 157)
        assert_raises_492349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 157)
        assert_raises_call_result_492352 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), assert_raises_492349, *[ValueError_492350], **kwargs_492351)
        
        with_492353 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 157, 13), assert_raises_call_result_492352, 'with parameter', '__enter__', '__exit__')

        if with_492353:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 157)
            enter___492354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 13), assert_raises_call_result_492352, '__enter__')
            with_enter_492355 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), enter___492354)
            
            # Assigning a Call to a Name (line 158):
            
            # Call to SphericalVoronoi(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'self' (line 158)
            self_492358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 52), 'self', False)
            # Obtaining the member 'points' of a type (line 158)
            points_492359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 52), self_492358, 'points')
            # Processing the call keyword arguments (line 158)
            float_492360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 59), 'float')
            keyword_492361 = float_492360
            kwargs_492362 = {'radius': keyword_492361}
            # Getting the type of 'spherical_voronoi' (line 158)
            spherical_voronoi_492356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'spherical_voronoi', False)
            # Obtaining the member 'SphericalVoronoi' of a type (line 158)
            SphericalVoronoi_492357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 17), spherical_voronoi_492356, 'SphericalVoronoi')
            # Calling SphericalVoronoi(args, kwargs) (line 158)
            SphericalVoronoi_call_result_492363 = invoke(stypy.reporting.localization.Localization(__file__, 158, 17), SphericalVoronoi_492357, *[points_492359], **kwargs_492362)
            
            # Assigning a type to the variable 'sv' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'sv', SphericalVoronoi_call_result_492363)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 157)
            exit___492364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 13), assert_raises_call_result_492352, '__exit__')
            with_exit_492365 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), exit___492364, None, None, None)

        
        # ################# End of 'test_incorrect_radius_handling(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_incorrect_radius_handling' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_492366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_incorrect_radius_handling'
        return stypy_return_type_492366


    @norecursion
    def test_incorrect_center_handling(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_incorrect_center_handling'
        module_type_store = module_type_store.open_function_context('test_incorrect_center_handling', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_function_name', 'TestSphericalVoronoi.test_incorrect_center_handling')
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalVoronoi.test_incorrect_center_handling.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.test_incorrect_center_handling', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_incorrect_center_handling', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_incorrect_center_handling(...)' code ##################

        
        # Call to assert_raises(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'ValueError' (line 164)
        ValueError_492368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 164)
        kwargs_492369 = {}
        # Getting the type of 'assert_raises' (line 164)
        assert_raises_492367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 164)
        assert_raises_call_result_492370 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), assert_raises_492367, *[ValueError_492368], **kwargs_492369)
        
        with_492371 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 164, 13), assert_raises_call_result_492370, 'with parameter', '__enter__', '__exit__')

        if with_492371:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 164)
            enter___492372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), assert_raises_call_result_492370, '__enter__')
            with_enter_492373 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), enter___492372)
            
            # Assigning a Call to a Name (line 165):
            
            # Call to SphericalVoronoi(...): (line 165)
            # Processing the call arguments (line 165)
            # Getting the type of 'self' (line 165)
            self_492376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 52), 'self', False)
            # Obtaining the member 'points' of a type (line 165)
            points_492377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 52), self_492376, 'points')
            # Processing the call keyword arguments (line 165)
            
            # Obtaining an instance of the builtin type 'list' (line 166)
            list_492378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 59), 'list')
            # Adding type elements to the builtin type 'list' instance (line 166)
            # Adding element type (line 166)
            float_492379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 60), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 59), list_492378, float_492379)
            # Adding element type (line 166)
            int_492380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 64), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 59), list_492378, int_492380)
            # Adding element type (line 166)
            int_492381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 66), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 59), list_492378, int_492381)
            
            keyword_492382 = list_492378
            kwargs_492383 = {'center': keyword_492382}
            # Getting the type of 'spherical_voronoi' (line 165)
            spherical_voronoi_492374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'spherical_voronoi', False)
            # Obtaining the member 'SphericalVoronoi' of a type (line 165)
            SphericalVoronoi_492375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 17), spherical_voronoi_492374, 'SphericalVoronoi')
            # Calling SphericalVoronoi(args, kwargs) (line 165)
            SphericalVoronoi_call_result_492384 = invoke(stypy.reporting.localization.Localization(__file__, 165, 17), SphericalVoronoi_492375, *[points_492377], **kwargs_492383)
            
            # Assigning a type to the variable 'sv' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'sv', SphericalVoronoi_call_result_492384)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 164)
            exit___492385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), assert_raises_call_result_492370, '__exit__')
            with_exit_492386 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), exit___492385, None, None, None)

        
        # ################# End of 'test_incorrect_center_handling(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_incorrect_center_handling' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_492387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_incorrect_center_handling'
        return stypy_return_type_492387


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 64, 0, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalVoronoi.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalVoronoi' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'TestSphericalVoronoi', TestSphericalVoronoi)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
