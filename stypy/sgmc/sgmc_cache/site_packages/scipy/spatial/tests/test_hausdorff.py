
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import numpy as np
4: from numpy.testing import (assert_almost_equal,
5:                            assert_array_equal,
6:                            assert_equal,
7:                            assert_)
8: from scipy.spatial.distance import directed_hausdorff
9: from scipy.spatial import distance
10: from scipy._lib._util import check_random_state
11: 
12: class TestHausdorff(object):
13:     # Test various properties of the directed Hausdorff code.
14: 
15:     def setup_method(self):
16:         np.random.seed(1234)
17:         random_angles = np.random.random(100) * np.pi * 2
18:         random_columns = np.column_stack(
19:             (random_angles, random_angles, np.zeros(100)))
20:         random_columns[..., 0] = np.cos(random_columns[..., 0])
21:         random_columns[..., 1] = np.sin(random_columns[..., 1])
22:         random_columns_2 = np.column_stack(
23:             (random_angles, random_angles, np.zeros(100)))
24:         random_columns_2[1:, 0] = np.cos(random_columns_2[1:, 0]) * 2.0
25:         random_columns_2[1:, 1] = np.sin(random_columns_2[1:, 1]) * 2.0
26:         # move one point farther out so we don't have two perfect circles
27:         random_columns_2[0, 0] = np.cos(random_columns_2[0, 0]) * 3.3
28:         random_columns_2[0, 1] = np.sin(random_columns_2[0, 1]) * 3.3
29:         self.path_1 = random_columns
30:         self.path_2 = random_columns_2
31:         self.path_1_4d = np.insert(self.path_1, 3, 5, axis=1)
32:         self.path_2_4d = np.insert(self.path_2, 3, 27, axis=1)
33: 
34:     def test_symmetry(self):
35:         # Ensure that the directed (asymmetric) Hausdorff distance is
36:         # actually asymmetric
37: 
38:         forward = directed_hausdorff(self.path_1, self.path_2)[0]
39:         reverse = directed_hausdorff(self.path_2, self.path_1)[0]
40:         assert_(forward != reverse)
41: 
42:     def test_brute_force_comparison_forward(self):
43:         # Ensure that the algorithm for directed_hausdorff gives the
44:         # same result as the simple / brute force approach in the
45:         # forward direction.
46:         actual = directed_hausdorff(self.path_1, self.path_2)[0]
47:         # brute force over rows:
48:         expected = max(np.amin(distance.cdist(self.path_1, self.path_2),
49:                                axis=1))
50:         assert_almost_equal(actual, expected, decimal=9)
51: 
52:     def test_brute_force_comparison_reverse(self):
53:         # Ensure that the algorithm for directed_hausdorff gives the
54:         # same result as the simple / brute force approach in the
55:         # reverse direction.
56:         actual = directed_hausdorff(self.path_2, self.path_1)[0]
57:         # brute force over columns:
58:         expected = max(np.amin(distance.cdist(self.path_1, self.path_2), 
59:                                axis=0))
60:         assert_almost_equal(actual, expected, decimal=9)
61: 
62:     def test_degenerate_case(self):
63:         # The directed Hausdorff distance must be zero if both input
64:         # data arrays match.
65:         actual = directed_hausdorff(self.path_1, self.path_1)[0]
66:         assert_almost_equal(actual, 0.0, decimal=9)
67: 
68:     def test_2d_data_forward(self):
69:         # Ensure that 2D data is handled properly for a simple case
70:         # relative to brute force approach.
71:         actual = directed_hausdorff(self.path_1[..., :2],
72:                                     self.path_2[..., :2])[0]
73:         expected = max(np.amin(distance.cdist(self.path_1[..., :2],
74:                                               self.path_2[..., :2]),
75:                                axis=1))
76:         assert_almost_equal(actual, expected, decimal=9)
77: 
78:     def test_4d_data_reverse(self):
79:         # Ensure that 4D data is handled properly for a simple case
80:         # relative to brute force approach.
81:         actual = directed_hausdorff(self.path_2_4d, self.path_1_4d)[0]
82:         # brute force over columns:
83:         expected = max(np.amin(distance.cdist(self.path_1_4d, self.path_2_4d), 
84:                                axis=0))
85:         assert_almost_equal(actual, expected, decimal=9)
86: 
87:     def test_indices(self):
88:         # Ensure that correct point indices are returned -- they should
89:         # correspond to the Hausdorff pair
90:         path_simple_1 = np.array([[-1,-12],[0,0], [1,1], [3,7], [1,2]])
91:         path_simple_2 = np.array([[0,0], [1,1], [4,100], [10,9]])
92:         actual = directed_hausdorff(path_simple_2, path_simple_1)[1:]
93:         expected = (2, 3)
94:         assert_array_equal(actual, expected)
95: 
96:     def test_random_state(self):
97:         # ensure that the global random state is not modified because
98:         # the directed Hausdorff algorithm uses randomization
99:         rs = check_random_state(None)
100:         old_global_state = rs.get_state()
101:         directed_hausdorff(self.path_1, self.path_2)
102:         rs2 = check_random_state(None)
103:         new_global_state = rs2.get_state()
104:         assert_equal(new_global_state, old_global_state)
105: 
106:     def test_random_state_None_int(self):
107:         # check that seed values of None or int do not alter global
108:         # random state
109:         for seed in [None, 27870671]:
110:             rs = check_random_state(None)
111:             old_global_state = rs.get_state()
112:             directed_hausdorff(self.path_1, self.path_2, seed)
113:             rs2 = check_random_state(None)
114:             new_global_state = rs2.get_state()
115:             assert_equal(new_global_state, old_global_state)
116: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_479848 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_479848) is not StypyTypeError):

    if (import_479848 != 'pyd_module'):
        __import__(import_479848)
        sys_modules_479849 = sys.modules[import_479848]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_479849.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_479848)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal, assert_' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_479850 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_479850) is not StypyTypeError):

    if (import_479850 != 'pyd_module'):
        __import__(import_479850)
        sys_modules_479851 = sys.modules[import_479850]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_479851.module_type_store, module_type_store, ['assert_almost_equal', 'assert_array_equal', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_479851, sys_modules_479851.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_array_equal', 'assert_equal', 'assert_'], [assert_almost_equal, assert_array_equal, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_479850)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.spatial.distance import directed_hausdorff' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_479852 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.spatial.distance')

if (type(import_479852) is not StypyTypeError):

    if (import_479852 != 'pyd_module'):
        __import__(import_479852)
        sys_modules_479853 = sys.modules[import_479852]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.spatial.distance', sys_modules_479853.module_type_store, module_type_store, ['directed_hausdorff'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_479853, sys_modules_479853.module_type_store, module_type_store)
    else:
        from scipy.spatial.distance import directed_hausdorff

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.spatial.distance', None, module_type_store, ['directed_hausdorff'], [directed_hausdorff])

else:
    # Assigning a type to the variable 'scipy.spatial.distance' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.spatial.distance', import_479852)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.spatial import distance' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_479854 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.spatial')

if (type(import_479854) is not StypyTypeError):

    if (import_479854 != 'pyd_module'):
        __import__(import_479854)
        sys_modules_479855 = sys.modules[import_479854]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.spatial', sys_modules_479855.module_type_store, module_type_store, ['distance'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_479855, sys_modules_479855.module_type_store, module_type_store)
    else:
        from scipy.spatial import distance

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.spatial', None, module_type_store, ['distance'], [distance])

else:
    # Assigning a type to the variable 'scipy.spatial' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.spatial', import_479854)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib._util import check_random_state' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_479856 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util')

if (type(import_479856) is not StypyTypeError):

    if (import_479856 != 'pyd_module'):
        __import__(import_479856)
        sys_modules_479857 = sys.modules[import_479856]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', sys_modules_479857.module_type_store, module_type_store, ['check_random_state'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_479857, sys_modules_479857.module_type_store, module_type_store)
    else:
        from scipy._lib._util import check_random_state

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', None, module_type_store, ['check_random_state'], [check_random_state])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', import_479856)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

# Declaration of the 'TestHausdorff' class

class TestHausdorff(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.setup_method')
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to seed(...): (line 16)
        # Processing the call arguments (line 16)
        int_479861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'int')
        # Processing the call keyword arguments (line 16)
        kwargs_479862 = {}
        # Getting the type of 'np' (line 16)
        np_479858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 16)
        random_479859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), np_479858, 'random')
        # Obtaining the member 'seed' of a type (line 16)
        seed_479860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), random_479859, 'seed')
        # Calling seed(args, kwargs) (line 16)
        seed_call_result_479863 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), seed_479860, *[int_479861], **kwargs_479862)
        
        
        # Assigning a BinOp to a Name (line 17):
        
        # Call to random(...): (line 17)
        # Processing the call arguments (line 17)
        int_479867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'int')
        # Processing the call keyword arguments (line 17)
        kwargs_479868 = {}
        # Getting the type of 'np' (line 17)
        np_479864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'np', False)
        # Obtaining the member 'random' of a type (line 17)
        random_479865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), np_479864, 'random')
        # Obtaining the member 'random' of a type (line 17)
        random_479866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), random_479865, 'random')
        # Calling random(args, kwargs) (line 17)
        random_call_result_479869 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), random_479866, *[int_479867], **kwargs_479868)
        
        # Getting the type of 'np' (line 17)
        np_479870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 48), 'np')
        # Obtaining the member 'pi' of a type (line 17)
        pi_479871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 48), np_479870, 'pi')
        # Applying the binary operator '*' (line 17)
        result_mul_479872 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 24), '*', random_call_result_479869, pi_479871)
        
        int_479873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 56), 'int')
        # Applying the binary operator '*' (line 17)
        result_mul_479874 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 54), '*', result_mul_479872, int_479873)
        
        # Assigning a type to the variable 'random_angles' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'random_angles', result_mul_479874)
        
        # Assigning a Call to a Name (line 18):
        
        # Call to column_stack(...): (line 18)
        # Processing the call arguments (line 18)
        
        # Obtaining an instance of the builtin type 'tuple' (line 19)
        tuple_479877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        # Adding element type (line 19)
        # Getting the type of 'random_angles' (line 19)
        random_angles_479878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'random_angles', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 13), tuple_479877, random_angles_479878)
        # Adding element type (line 19)
        # Getting the type of 'random_angles' (line 19)
        random_angles_479879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), 'random_angles', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 13), tuple_479877, random_angles_479879)
        # Adding element type (line 19)
        
        # Call to zeros(...): (line 19)
        # Processing the call arguments (line 19)
        int_479882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 52), 'int')
        # Processing the call keyword arguments (line 19)
        kwargs_479883 = {}
        # Getting the type of 'np' (line 19)
        np_479880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 43), 'np', False)
        # Obtaining the member 'zeros' of a type (line 19)
        zeros_479881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 43), np_479880, 'zeros')
        # Calling zeros(args, kwargs) (line 19)
        zeros_call_result_479884 = invoke(stypy.reporting.localization.Localization(__file__, 19, 43), zeros_479881, *[int_479882], **kwargs_479883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 13), tuple_479877, zeros_call_result_479884)
        
        # Processing the call keyword arguments (line 18)
        kwargs_479885 = {}
        # Getting the type of 'np' (line 18)
        np_479875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 18)
        column_stack_479876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), np_479875, 'column_stack')
        # Calling column_stack(args, kwargs) (line 18)
        column_stack_call_result_479886 = invoke(stypy.reporting.localization.Localization(__file__, 18, 25), column_stack_479876, *[tuple_479877], **kwargs_479885)
        
        # Assigning a type to the variable 'random_columns' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'random_columns', column_stack_call_result_479886)
        
        # Assigning a Call to a Subscript (line 20):
        
        # Call to cos(...): (line 20)
        # Processing the call arguments (line 20)
        
        # Obtaining the type of the subscript
        Ellipsis_479889 = Ellipsis
        int_479890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 60), 'int')
        # Getting the type of 'random_columns' (line 20)
        random_columns_479891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 40), 'random_columns', False)
        # Obtaining the member '__getitem__' of a type (line 20)
        getitem___479892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 40), random_columns_479891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 20)
        subscript_call_result_479893 = invoke(stypy.reporting.localization.Localization(__file__, 20, 40), getitem___479892, (Ellipsis_479889, int_479890))
        
        # Processing the call keyword arguments (line 20)
        kwargs_479894 = {}
        # Getting the type of 'np' (line 20)
        np_479887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'np', False)
        # Obtaining the member 'cos' of a type (line 20)
        cos_479888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 33), np_479887, 'cos')
        # Calling cos(args, kwargs) (line 20)
        cos_call_result_479895 = invoke(stypy.reporting.localization.Localization(__file__, 20, 33), cos_479888, *[subscript_call_result_479893], **kwargs_479894)
        
        # Getting the type of 'random_columns' (line 20)
        random_columns_479896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'random_columns')
        Ellipsis_479897 = Ellipsis
        int_479898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'int')
        # Storing an element on a container (line 20)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), random_columns_479896, ((Ellipsis_479897, int_479898), cos_call_result_479895))
        
        # Assigning a Call to a Subscript (line 21):
        
        # Call to sin(...): (line 21)
        # Processing the call arguments (line 21)
        
        # Obtaining the type of the subscript
        Ellipsis_479901 = Ellipsis
        int_479902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 60), 'int')
        # Getting the type of 'random_columns' (line 21)
        random_columns_479903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 40), 'random_columns', False)
        # Obtaining the member '__getitem__' of a type (line 21)
        getitem___479904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 40), random_columns_479903, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 21)
        subscript_call_result_479905 = invoke(stypy.reporting.localization.Localization(__file__, 21, 40), getitem___479904, (Ellipsis_479901, int_479902))
        
        # Processing the call keyword arguments (line 21)
        kwargs_479906 = {}
        # Getting the type of 'np' (line 21)
        np_479899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 33), 'np', False)
        # Obtaining the member 'sin' of a type (line 21)
        sin_479900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 33), np_479899, 'sin')
        # Calling sin(args, kwargs) (line 21)
        sin_call_result_479907 = invoke(stypy.reporting.localization.Localization(__file__, 21, 33), sin_479900, *[subscript_call_result_479905], **kwargs_479906)
        
        # Getting the type of 'random_columns' (line 21)
        random_columns_479908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'random_columns')
        Ellipsis_479909 = Ellipsis
        int_479910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
        # Storing an element on a container (line 21)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), random_columns_479908, ((Ellipsis_479909, int_479910), sin_call_result_479907))
        
        # Assigning a Call to a Name (line 22):
        
        # Call to column_stack(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_479913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        # Getting the type of 'random_angles' (line 23)
        random_angles_479914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'random_angles', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), tuple_479913, random_angles_479914)
        # Adding element type (line 23)
        # Getting the type of 'random_angles' (line 23)
        random_angles_479915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 28), 'random_angles', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), tuple_479913, random_angles_479915)
        # Adding element type (line 23)
        
        # Call to zeros(...): (line 23)
        # Processing the call arguments (line 23)
        int_479918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 52), 'int')
        # Processing the call keyword arguments (line 23)
        kwargs_479919 = {}
        # Getting the type of 'np' (line 23)
        np_479916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 43), 'np', False)
        # Obtaining the member 'zeros' of a type (line 23)
        zeros_479917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 43), np_479916, 'zeros')
        # Calling zeros(args, kwargs) (line 23)
        zeros_call_result_479920 = invoke(stypy.reporting.localization.Localization(__file__, 23, 43), zeros_479917, *[int_479918], **kwargs_479919)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), tuple_479913, zeros_call_result_479920)
        
        # Processing the call keyword arguments (line 22)
        kwargs_479921 = {}
        # Getting the type of 'np' (line 22)
        np_479911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 22)
        column_stack_479912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 27), np_479911, 'column_stack')
        # Calling column_stack(args, kwargs) (line 22)
        column_stack_call_result_479922 = invoke(stypy.reporting.localization.Localization(__file__, 22, 27), column_stack_479912, *[tuple_479913], **kwargs_479921)
        
        # Assigning a type to the variable 'random_columns_2' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'random_columns_2', column_stack_call_result_479922)
        
        # Assigning a BinOp to a Subscript (line 24):
        
        # Call to cos(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining the type of the subscript
        int_479925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 58), 'int')
        slice_479926 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 41), int_479925, None, None)
        int_479927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 62), 'int')
        # Getting the type of 'random_columns_2' (line 24)
        random_columns_2_479928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'random_columns_2', False)
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___479929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 41), random_columns_2_479928, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_479930 = invoke(stypy.reporting.localization.Localization(__file__, 24, 41), getitem___479929, (slice_479926, int_479927))
        
        # Processing the call keyword arguments (line 24)
        kwargs_479931 = {}
        # Getting the type of 'np' (line 24)
        np_479923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), 'np', False)
        # Obtaining the member 'cos' of a type (line 24)
        cos_479924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 34), np_479923, 'cos')
        # Calling cos(args, kwargs) (line 24)
        cos_call_result_479932 = invoke(stypy.reporting.localization.Localization(__file__, 24, 34), cos_479924, *[subscript_call_result_479930], **kwargs_479931)
        
        float_479933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 68), 'float')
        # Applying the binary operator '*' (line 24)
        result_mul_479934 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 34), '*', cos_call_result_479932, float_479933)
        
        # Getting the type of 'random_columns_2' (line 24)
        random_columns_2_479935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'random_columns_2')
        int_479936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
        slice_479937 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 8), int_479936, None, None)
        int_479938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'int')
        # Storing an element on a container (line 24)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), random_columns_2_479935, ((slice_479937, int_479938), result_mul_479934))
        
        # Assigning a BinOp to a Subscript (line 25):
        
        # Call to sin(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Obtaining the type of the subscript
        int_479941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 58), 'int')
        slice_479942 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 41), int_479941, None, None)
        int_479943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 62), 'int')
        # Getting the type of 'random_columns_2' (line 25)
        random_columns_2_479944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 41), 'random_columns_2', False)
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___479945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 41), random_columns_2_479944, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_479946 = invoke(stypy.reporting.localization.Localization(__file__, 25, 41), getitem___479945, (slice_479942, int_479943))
        
        # Processing the call keyword arguments (line 25)
        kwargs_479947 = {}
        # Getting the type of 'np' (line 25)
        np_479939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'np', False)
        # Obtaining the member 'sin' of a type (line 25)
        sin_479940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 34), np_479939, 'sin')
        # Calling sin(args, kwargs) (line 25)
        sin_call_result_479948 = invoke(stypy.reporting.localization.Localization(__file__, 25, 34), sin_479940, *[subscript_call_result_479946], **kwargs_479947)
        
        float_479949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 68), 'float')
        # Applying the binary operator '*' (line 25)
        result_mul_479950 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 34), '*', sin_call_result_479948, float_479949)
        
        # Getting the type of 'random_columns_2' (line 25)
        random_columns_2_479951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'random_columns_2')
        int_479952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
        slice_479953 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 8), int_479952, None, None)
        int_479954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'int')
        # Storing an element on a container (line 25)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), random_columns_2_479951, ((slice_479953, int_479954), result_mul_479950))
        
        # Assigning a BinOp to a Subscript (line 27):
        
        # Call to cos(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_479957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        int_479958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 57), tuple_479957, int_479958)
        # Adding element type (line 27)
        int_479959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 57), tuple_479957, int_479959)
        
        # Getting the type of 'random_columns_2' (line 27)
        random_columns_2_479960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 40), 'random_columns_2', False)
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___479961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 40), random_columns_2_479960, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_479962 = invoke(stypy.reporting.localization.Localization(__file__, 27, 40), getitem___479961, tuple_479957)
        
        # Processing the call keyword arguments (line 27)
        kwargs_479963 = {}
        # Getting the type of 'np' (line 27)
        np_479955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'np', False)
        # Obtaining the member 'cos' of a type (line 27)
        cos_479956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 33), np_479955, 'cos')
        # Calling cos(args, kwargs) (line 27)
        cos_call_result_479964 = invoke(stypy.reporting.localization.Localization(__file__, 27, 33), cos_479956, *[subscript_call_result_479962], **kwargs_479963)
        
        float_479965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 66), 'float')
        # Applying the binary operator '*' (line 27)
        result_mul_479966 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 33), '*', cos_call_result_479964, float_479965)
        
        # Getting the type of 'random_columns_2' (line 27)
        random_columns_2_479967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'random_columns_2')
        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_479968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        int_479969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 25), tuple_479968, int_479969)
        # Adding element type (line 27)
        int_479970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 25), tuple_479968, int_479970)
        
        # Storing an element on a container (line 27)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 8), random_columns_2_479967, (tuple_479968, result_mul_479966))
        
        # Assigning a BinOp to a Subscript (line 28):
        
        # Call to sin(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_479973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        int_479974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 57), tuple_479973, int_479974)
        # Adding element type (line 28)
        int_479975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 57), tuple_479973, int_479975)
        
        # Getting the type of 'random_columns_2' (line 28)
        random_columns_2_479976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'random_columns_2', False)
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___479977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 40), random_columns_2_479976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_479978 = invoke(stypy.reporting.localization.Localization(__file__, 28, 40), getitem___479977, tuple_479973)
        
        # Processing the call keyword arguments (line 28)
        kwargs_479979 = {}
        # Getting the type of 'np' (line 28)
        np_479971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 33), 'np', False)
        # Obtaining the member 'sin' of a type (line 28)
        sin_479972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 33), np_479971, 'sin')
        # Calling sin(args, kwargs) (line 28)
        sin_call_result_479980 = invoke(stypy.reporting.localization.Localization(__file__, 28, 33), sin_479972, *[subscript_call_result_479978], **kwargs_479979)
        
        float_479981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 66), 'float')
        # Applying the binary operator '*' (line 28)
        result_mul_479982 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 33), '*', sin_call_result_479980, float_479981)
        
        # Getting the type of 'random_columns_2' (line 28)
        random_columns_2_479983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'random_columns_2')
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_479984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        int_479985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 25), tuple_479984, int_479985)
        # Adding element type (line 28)
        int_479986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 25), tuple_479984, int_479986)
        
        # Storing an element on a container (line 28)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), random_columns_2_479983, (tuple_479984, result_mul_479982))
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'random_columns' (line 29)
        random_columns_479987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'random_columns')
        # Getting the type of 'self' (line 29)
        self_479988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'path_1' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_479988, 'path_1', random_columns_479987)
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'random_columns_2' (line 30)
        random_columns_2_479989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'random_columns_2')
        # Getting the type of 'self' (line 30)
        self_479990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'path_2' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_479990, 'path_2', random_columns_2_479989)
        
        # Assigning a Call to a Attribute (line 31):
        
        # Call to insert(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'self' (line 31)
        self_479993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 35), 'self', False)
        # Obtaining the member 'path_1' of a type (line 31)
        path_1_479994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 35), self_479993, 'path_1')
        int_479995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 48), 'int')
        int_479996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 51), 'int')
        # Processing the call keyword arguments (line 31)
        int_479997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 59), 'int')
        keyword_479998 = int_479997
        kwargs_479999 = {'axis': keyword_479998}
        # Getting the type of 'np' (line 31)
        np_479991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'np', False)
        # Obtaining the member 'insert' of a type (line 31)
        insert_479992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 25), np_479991, 'insert')
        # Calling insert(args, kwargs) (line 31)
        insert_call_result_480000 = invoke(stypy.reporting.localization.Localization(__file__, 31, 25), insert_479992, *[path_1_479994, int_479995, int_479996], **kwargs_479999)
        
        # Getting the type of 'self' (line 31)
        self_480001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'path_1_4d' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_480001, 'path_1_4d', insert_call_result_480000)
        
        # Assigning a Call to a Attribute (line 32):
        
        # Call to insert(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'self' (line 32)
        self_480004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'self', False)
        # Obtaining the member 'path_2' of a type (line 32)
        path_2_480005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 35), self_480004, 'path_2')
        int_480006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 48), 'int')
        int_480007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 51), 'int')
        # Processing the call keyword arguments (line 32)
        int_480008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 60), 'int')
        keyword_480009 = int_480008
        kwargs_480010 = {'axis': keyword_480009}
        # Getting the type of 'np' (line 32)
        np_480002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'np', False)
        # Obtaining the member 'insert' of a type (line 32)
        insert_480003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), np_480002, 'insert')
        # Calling insert(args, kwargs) (line 32)
        insert_call_result_480011 = invoke(stypy.reporting.localization.Localization(__file__, 32, 25), insert_480003, *[path_2_480005, int_480006, int_480007], **kwargs_480010)
        
        # Getting the type of 'self' (line 32)
        self_480012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'path_2_4d' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_480012, 'path_2_4d', insert_call_result_480011)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_480013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480013)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_480013


    @norecursion
    def test_symmetry(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_symmetry'
        module_type_store = module_type_store.open_function_context('test_symmetry', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_symmetry')
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_symmetry.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_symmetry', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_symmetry', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_symmetry(...)' code ##################

        
        # Assigning a Subscript to a Name (line 38):
        
        # Obtaining the type of the subscript
        int_480014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 63), 'int')
        
        # Call to directed_hausdorff(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_480016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 37), 'self', False)
        # Obtaining the member 'path_1' of a type (line 38)
        path_1_480017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 37), self_480016, 'path_1')
        # Getting the type of 'self' (line 38)
        self_480018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 50), 'self', False)
        # Obtaining the member 'path_2' of a type (line 38)
        path_2_480019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 50), self_480018, 'path_2')
        # Processing the call keyword arguments (line 38)
        kwargs_480020 = {}
        # Getting the type of 'directed_hausdorff' (line 38)
        directed_hausdorff_480015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 38)
        directed_hausdorff_call_result_480021 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), directed_hausdorff_480015, *[path_1_480017, path_2_480019], **kwargs_480020)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___480022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 18), directed_hausdorff_call_result_480021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_480023 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), getitem___480022, int_480014)
        
        # Assigning a type to the variable 'forward' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'forward', subscript_call_result_480023)
        
        # Assigning a Subscript to a Name (line 39):
        
        # Obtaining the type of the subscript
        int_480024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 63), 'int')
        
        # Call to directed_hausdorff(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_480026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'self', False)
        # Obtaining the member 'path_2' of a type (line 39)
        path_2_480027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 37), self_480026, 'path_2')
        # Getting the type of 'self' (line 39)
        self_480028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 50), 'self', False)
        # Obtaining the member 'path_1' of a type (line 39)
        path_1_480029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 50), self_480028, 'path_1')
        # Processing the call keyword arguments (line 39)
        kwargs_480030 = {}
        # Getting the type of 'directed_hausdorff' (line 39)
        directed_hausdorff_480025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 39)
        directed_hausdorff_call_result_480031 = invoke(stypy.reporting.localization.Localization(__file__, 39, 18), directed_hausdorff_480025, *[path_2_480027, path_1_480029], **kwargs_480030)
        
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___480032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 18), directed_hausdorff_call_result_480031, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_480033 = invoke(stypy.reporting.localization.Localization(__file__, 39, 18), getitem___480032, int_480024)
        
        # Assigning a type to the variable 'reverse' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'reverse', subscript_call_result_480033)
        
        # Call to assert_(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Getting the type of 'forward' (line 40)
        forward_480035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'forward', False)
        # Getting the type of 'reverse' (line 40)
        reverse_480036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'reverse', False)
        # Applying the binary operator '!=' (line 40)
        result_ne_480037 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), '!=', forward_480035, reverse_480036)
        
        # Processing the call keyword arguments (line 40)
        kwargs_480038 = {}
        # Getting the type of 'assert_' (line 40)
        assert__480034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 40)
        assert__call_result_480039 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assert__480034, *[result_ne_480037], **kwargs_480038)
        
        
        # ################# End of 'test_symmetry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_symmetry' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_480040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480040)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_symmetry'
        return stypy_return_type_480040


    @norecursion
    def test_brute_force_comparison_forward(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_brute_force_comparison_forward'
        module_type_store = module_type_store.open_function_context('test_brute_force_comparison_forward', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_brute_force_comparison_forward')
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_brute_force_comparison_forward.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_brute_force_comparison_forward', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_brute_force_comparison_forward', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_brute_force_comparison_forward(...)' code ##################

        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        int_480041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 62), 'int')
        
        # Call to directed_hausdorff(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_480043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'self', False)
        # Obtaining the member 'path_1' of a type (line 46)
        path_1_480044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 36), self_480043, 'path_1')
        # Getting the type of 'self' (line 46)
        self_480045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 49), 'self', False)
        # Obtaining the member 'path_2' of a type (line 46)
        path_2_480046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 49), self_480045, 'path_2')
        # Processing the call keyword arguments (line 46)
        kwargs_480047 = {}
        # Getting the type of 'directed_hausdorff' (line 46)
        directed_hausdorff_480042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 46)
        directed_hausdorff_call_result_480048 = invoke(stypy.reporting.localization.Localization(__file__, 46, 17), directed_hausdorff_480042, *[path_1_480044, path_2_480046], **kwargs_480047)
        
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___480049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 17), directed_hausdorff_call_result_480048, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_480050 = invoke(stypy.reporting.localization.Localization(__file__, 46, 17), getitem___480049, int_480041)
        
        # Assigning a type to the variable 'actual' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'actual', subscript_call_result_480050)
        
        # Assigning a Call to a Name (line 48):
        
        # Call to max(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to amin(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to cdist(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'self' (line 48)
        self_480056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 46), 'self', False)
        # Obtaining the member 'path_1' of a type (line 48)
        path_1_480057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 46), self_480056, 'path_1')
        # Getting the type of 'self' (line 48)
        self_480058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 59), 'self', False)
        # Obtaining the member 'path_2' of a type (line 48)
        path_2_480059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 59), self_480058, 'path_2')
        # Processing the call keyword arguments (line 48)
        kwargs_480060 = {}
        # Getting the type of 'distance' (line 48)
        distance_480054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'distance', False)
        # Obtaining the member 'cdist' of a type (line 48)
        cdist_480055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 31), distance_480054, 'cdist')
        # Calling cdist(args, kwargs) (line 48)
        cdist_call_result_480061 = invoke(stypy.reporting.localization.Localization(__file__, 48, 31), cdist_480055, *[path_1_480057, path_2_480059], **kwargs_480060)
        
        # Processing the call keyword arguments (line 48)
        int_480062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'int')
        keyword_480063 = int_480062
        kwargs_480064 = {'axis': keyword_480063}
        # Getting the type of 'np' (line 48)
        np_480052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'np', False)
        # Obtaining the member 'amin' of a type (line 48)
        amin_480053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 23), np_480052, 'amin')
        # Calling amin(args, kwargs) (line 48)
        amin_call_result_480065 = invoke(stypy.reporting.localization.Localization(__file__, 48, 23), amin_480053, *[cdist_call_result_480061], **kwargs_480064)
        
        # Processing the call keyword arguments (line 48)
        kwargs_480066 = {}
        # Getting the type of 'max' (line 48)
        max_480051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'max', False)
        # Calling max(args, kwargs) (line 48)
        max_call_result_480067 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), max_480051, *[amin_call_result_480065], **kwargs_480066)
        
        # Assigning a type to the variable 'expected' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'expected', max_call_result_480067)
        
        # Call to assert_almost_equal(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'actual' (line 50)
        actual_480069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'actual', False)
        # Getting the type of 'expected' (line 50)
        expected_480070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'expected', False)
        # Processing the call keyword arguments (line 50)
        int_480071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 54), 'int')
        keyword_480072 = int_480071
        kwargs_480073 = {'decimal': keyword_480072}
        # Getting the type of 'assert_almost_equal' (line 50)
        assert_almost_equal_480068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 50)
        assert_almost_equal_call_result_480074 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert_almost_equal_480068, *[actual_480069, expected_480070], **kwargs_480073)
        
        
        # ################# End of 'test_brute_force_comparison_forward(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_brute_force_comparison_forward' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_480075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480075)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_brute_force_comparison_forward'
        return stypy_return_type_480075


    @norecursion
    def test_brute_force_comparison_reverse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_brute_force_comparison_reverse'
        module_type_store = module_type_store.open_function_context('test_brute_force_comparison_reverse', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_brute_force_comparison_reverse')
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_brute_force_comparison_reverse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_brute_force_comparison_reverse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_brute_force_comparison_reverse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_brute_force_comparison_reverse(...)' code ##################

        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_480076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 62), 'int')
        
        # Call to directed_hausdorff(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'self' (line 56)
        self_480078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'self', False)
        # Obtaining the member 'path_2' of a type (line 56)
        path_2_480079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 36), self_480078, 'path_2')
        # Getting the type of 'self' (line 56)
        self_480080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 49), 'self', False)
        # Obtaining the member 'path_1' of a type (line 56)
        path_1_480081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 49), self_480080, 'path_1')
        # Processing the call keyword arguments (line 56)
        kwargs_480082 = {}
        # Getting the type of 'directed_hausdorff' (line 56)
        directed_hausdorff_480077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 56)
        directed_hausdorff_call_result_480083 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), directed_hausdorff_480077, *[path_2_480079, path_1_480081], **kwargs_480082)
        
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___480084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), directed_hausdorff_call_result_480083, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_480085 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), getitem___480084, int_480076)
        
        # Assigning a type to the variable 'actual' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'actual', subscript_call_result_480085)
        
        # Assigning a Call to a Name (line 58):
        
        # Call to max(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to amin(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to cdist(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'self' (line 58)
        self_480091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 46), 'self', False)
        # Obtaining the member 'path_1' of a type (line 58)
        path_1_480092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 46), self_480091, 'path_1')
        # Getting the type of 'self' (line 58)
        self_480093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 59), 'self', False)
        # Obtaining the member 'path_2' of a type (line 58)
        path_2_480094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 59), self_480093, 'path_2')
        # Processing the call keyword arguments (line 58)
        kwargs_480095 = {}
        # Getting the type of 'distance' (line 58)
        distance_480089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'distance', False)
        # Obtaining the member 'cdist' of a type (line 58)
        cdist_480090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 31), distance_480089, 'cdist')
        # Calling cdist(args, kwargs) (line 58)
        cdist_call_result_480096 = invoke(stypy.reporting.localization.Localization(__file__, 58, 31), cdist_480090, *[path_1_480092, path_2_480094], **kwargs_480095)
        
        # Processing the call keyword arguments (line 58)
        int_480097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'int')
        keyword_480098 = int_480097
        kwargs_480099 = {'axis': keyword_480098}
        # Getting the type of 'np' (line 58)
        np_480087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'np', False)
        # Obtaining the member 'amin' of a type (line 58)
        amin_480088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 23), np_480087, 'amin')
        # Calling amin(args, kwargs) (line 58)
        amin_call_result_480100 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), amin_480088, *[cdist_call_result_480096], **kwargs_480099)
        
        # Processing the call keyword arguments (line 58)
        kwargs_480101 = {}
        # Getting the type of 'max' (line 58)
        max_480086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'max', False)
        # Calling max(args, kwargs) (line 58)
        max_call_result_480102 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), max_480086, *[amin_call_result_480100], **kwargs_480101)
        
        # Assigning a type to the variable 'expected' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'expected', max_call_result_480102)
        
        # Call to assert_almost_equal(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'actual' (line 60)
        actual_480104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 28), 'actual', False)
        # Getting the type of 'expected' (line 60)
        expected_480105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'expected', False)
        # Processing the call keyword arguments (line 60)
        int_480106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 54), 'int')
        keyword_480107 = int_480106
        kwargs_480108 = {'decimal': keyword_480107}
        # Getting the type of 'assert_almost_equal' (line 60)
        assert_almost_equal_480103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 60)
        assert_almost_equal_call_result_480109 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_almost_equal_480103, *[actual_480104, expected_480105], **kwargs_480108)
        
        
        # ################# End of 'test_brute_force_comparison_reverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_brute_force_comparison_reverse' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_480110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480110)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_brute_force_comparison_reverse'
        return stypy_return_type_480110


    @norecursion
    def test_degenerate_case(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_degenerate_case'
        module_type_store = module_type_store.open_function_context('test_degenerate_case', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_degenerate_case')
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_degenerate_case.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_degenerate_case', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_degenerate_case', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_degenerate_case(...)' code ##################

        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_480111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 62), 'int')
        
        # Call to directed_hausdorff(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_480113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'self', False)
        # Obtaining the member 'path_1' of a type (line 65)
        path_1_480114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 36), self_480113, 'path_1')
        # Getting the type of 'self' (line 65)
        self_480115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 49), 'self', False)
        # Obtaining the member 'path_1' of a type (line 65)
        path_1_480116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 49), self_480115, 'path_1')
        # Processing the call keyword arguments (line 65)
        kwargs_480117 = {}
        # Getting the type of 'directed_hausdorff' (line 65)
        directed_hausdorff_480112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 65)
        directed_hausdorff_call_result_480118 = invoke(stypy.reporting.localization.Localization(__file__, 65, 17), directed_hausdorff_480112, *[path_1_480114, path_1_480116], **kwargs_480117)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___480119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 17), directed_hausdorff_call_result_480118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_480120 = invoke(stypy.reporting.localization.Localization(__file__, 65, 17), getitem___480119, int_480111)
        
        # Assigning a type to the variable 'actual' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'actual', subscript_call_result_480120)
        
        # Call to assert_almost_equal(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'actual' (line 66)
        actual_480122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'actual', False)
        float_480123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 36), 'float')
        # Processing the call keyword arguments (line 66)
        int_480124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'int')
        keyword_480125 = int_480124
        kwargs_480126 = {'decimal': keyword_480125}
        # Getting the type of 'assert_almost_equal' (line 66)
        assert_almost_equal_480121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 66)
        assert_almost_equal_call_result_480127 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_almost_equal_480121, *[actual_480122, float_480123], **kwargs_480126)
        
        
        # ################# End of 'test_degenerate_case(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_degenerate_case' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_480128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_degenerate_case'
        return stypy_return_type_480128


    @norecursion
    def test_2d_data_forward(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2d_data_forward'
        module_type_store = module_type_store.open_function_context('test_2d_data_forward', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_2d_data_forward')
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_2d_data_forward.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_2d_data_forward', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2d_data_forward', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2d_data_forward(...)' code ##################

        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_480129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 58), 'int')
        
        # Call to directed_hausdorff(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining the type of the subscript
        Ellipsis_480131 = Ellipsis
        int_480132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 54), 'int')
        slice_480133 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 36), None, int_480132, None)
        # Getting the type of 'self' (line 71)
        self_480134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 36), 'self', False)
        # Obtaining the member 'path_1' of a type (line 71)
        path_1_480135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 36), self_480134, 'path_1')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___480136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 36), path_1_480135, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_480137 = invoke(stypy.reporting.localization.Localization(__file__, 71, 36), getitem___480136, (Ellipsis_480131, slice_480133))
        
        
        # Obtaining the type of the subscript
        Ellipsis_480138 = Ellipsis
        int_480139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 54), 'int')
        slice_480140 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 36), None, int_480139, None)
        # Getting the type of 'self' (line 72)
        self_480141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'self', False)
        # Obtaining the member 'path_2' of a type (line 72)
        path_2_480142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 36), self_480141, 'path_2')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___480143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 36), path_2_480142, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_480144 = invoke(stypy.reporting.localization.Localization(__file__, 72, 36), getitem___480143, (Ellipsis_480138, slice_480140))
        
        # Processing the call keyword arguments (line 71)
        kwargs_480145 = {}
        # Getting the type of 'directed_hausdorff' (line 71)
        directed_hausdorff_480130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 71)
        directed_hausdorff_call_result_480146 = invoke(stypy.reporting.localization.Localization(__file__, 71, 17), directed_hausdorff_480130, *[subscript_call_result_480137, subscript_call_result_480144], **kwargs_480145)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___480147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 17), directed_hausdorff_call_result_480146, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_480148 = invoke(stypy.reporting.localization.Localization(__file__, 71, 17), getitem___480147, int_480129)
        
        # Assigning a type to the variable 'actual' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'actual', subscript_call_result_480148)
        
        # Assigning a Call to a Name (line 73):
        
        # Call to max(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to amin(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to cdist(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Obtaining the type of the subscript
        Ellipsis_480154 = Ellipsis
        int_480155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 64), 'int')
        slice_480156 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 73, 46), None, int_480155, None)
        # Getting the type of 'self' (line 73)
        self_480157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 46), 'self', False)
        # Obtaining the member 'path_1' of a type (line 73)
        path_1_480158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 46), self_480157, 'path_1')
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___480159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 46), path_1_480158, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_480160 = invoke(stypy.reporting.localization.Localization(__file__, 73, 46), getitem___480159, (Ellipsis_480154, slice_480156))
        
        
        # Obtaining the type of the subscript
        Ellipsis_480161 = Ellipsis
        int_480162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 64), 'int')
        slice_480163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 74, 46), None, int_480162, None)
        # Getting the type of 'self' (line 74)
        self_480164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 46), 'self', False)
        # Obtaining the member 'path_2' of a type (line 74)
        path_2_480165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 46), self_480164, 'path_2')
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___480166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 46), path_2_480165, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_480167 = invoke(stypy.reporting.localization.Localization(__file__, 74, 46), getitem___480166, (Ellipsis_480161, slice_480163))
        
        # Processing the call keyword arguments (line 73)
        kwargs_480168 = {}
        # Getting the type of 'distance' (line 73)
        distance_480152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'distance', False)
        # Obtaining the member 'cdist' of a type (line 73)
        cdist_480153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 31), distance_480152, 'cdist')
        # Calling cdist(args, kwargs) (line 73)
        cdist_call_result_480169 = invoke(stypy.reporting.localization.Localization(__file__, 73, 31), cdist_480153, *[subscript_call_result_480160, subscript_call_result_480167], **kwargs_480168)
        
        # Processing the call keyword arguments (line 73)
        int_480170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 36), 'int')
        keyword_480171 = int_480170
        kwargs_480172 = {'axis': keyword_480171}
        # Getting the type of 'np' (line 73)
        np_480150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'np', False)
        # Obtaining the member 'amin' of a type (line 73)
        amin_480151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), np_480150, 'amin')
        # Calling amin(args, kwargs) (line 73)
        amin_call_result_480173 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), amin_480151, *[cdist_call_result_480169], **kwargs_480172)
        
        # Processing the call keyword arguments (line 73)
        kwargs_480174 = {}
        # Getting the type of 'max' (line 73)
        max_480149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'max', False)
        # Calling max(args, kwargs) (line 73)
        max_call_result_480175 = invoke(stypy.reporting.localization.Localization(__file__, 73, 19), max_480149, *[amin_call_result_480173], **kwargs_480174)
        
        # Assigning a type to the variable 'expected' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'expected', max_call_result_480175)
        
        # Call to assert_almost_equal(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'actual' (line 76)
        actual_480177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'actual', False)
        # Getting the type of 'expected' (line 76)
        expected_480178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'expected', False)
        # Processing the call keyword arguments (line 76)
        int_480179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 54), 'int')
        keyword_480180 = int_480179
        kwargs_480181 = {'decimal': keyword_480180}
        # Getting the type of 'assert_almost_equal' (line 76)
        assert_almost_equal_480176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 76)
        assert_almost_equal_call_result_480182 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_almost_equal_480176, *[actual_480177, expected_480178], **kwargs_480181)
        
        
        # ################# End of 'test_2d_data_forward(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2d_data_forward' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_480183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480183)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2d_data_forward'
        return stypy_return_type_480183


    @norecursion
    def test_4d_data_reverse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_4d_data_reverse'
        module_type_store = module_type_store.open_function_context('test_4d_data_reverse', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_4d_data_reverse')
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_4d_data_reverse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_4d_data_reverse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_4d_data_reverse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_4d_data_reverse(...)' code ##################

        
        # Assigning a Subscript to a Name (line 81):
        
        # Obtaining the type of the subscript
        int_480184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 68), 'int')
        
        # Call to directed_hausdorff(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'self' (line 81)
        self_480186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 36), 'self', False)
        # Obtaining the member 'path_2_4d' of a type (line 81)
        path_2_4d_480187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 36), self_480186, 'path_2_4d')
        # Getting the type of 'self' (line 81)
        self_480188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 52), 'self', False)
        # Obtaining the member 'path_1_4d' of a type (line 81)
        path_1_4d_480189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 52), self_480188, 'path_1_4d')
        # Processing the call keyword arguments (line 81)
        kwargs_480190 = {}
        # Getting the type of 'directed_hausdorff' (line 81)
        directed_hausdorff_480185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 81)
        directed_hausdorff_call_result_480191 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), directed_hausdorff_480185, *[path_2_4d_480187, path_1_4d_480189], **kwargs_480190)
        
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___480192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), directed_hausdorff_call_result_480191, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_480193 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), getitem___480192, int_480184)
        
        # Assigning a type to the variable 'actual' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'actual', subscript_call_result_480193)
        
        # Assigning a Call to a Name (line 83):
        
        # Call to max(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to amin(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to cdist(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'self' (line 83)
        self_480199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'self', False)
        # Obtaining the member 'path_1_4d' of a type (line 83)
        path_1_4d_480200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 46), self_480199, 'path_1_4d')
        # Getting the type of 'self' (line 83)
        self_480201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 62), 'self', False)
        # Obtaining the member 'path_2_4d' of a type (line 83)
        path_2_4d_480202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 62), self_480201, 'path_2_4d')
        # Processing the call keyword arguments (line 83)
        kwargs_480203 = {}
        # Getting the type of 'distance' (line 83)
        distance_480197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'distance', False)
        # Obtaining the member 'cdist' of a type (line 83)
        cdist_480198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 31), distance_480197, 'cdist')
        # Calling cdist(args, kwargs) (line 83)
        cdist_call_result_480204 = invoke(stypy.reporting.localization.Localization(__file__, 83, 31), cdist_480198, *[path_1_4d_480200, path_2_4d_480202], **kwargs_480203)
        
        # Processing the call keyword arguments (line 83)
        int_480205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 36), 'int')
        keyword_480206 = int_480205
        kwargs_480207 = {'axis': keyword_480206}
        # Getting the type of 'np' (line 83)
        np_480195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'np', False)
        # Obtaining the member 'amin' of a type (line 83)
        amin_480196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 23), np_480195, 'amin')
        # Calling amin(args, kwargs) (line 83)
        amin_call_result_480208 = invoke(stypy.reporting.localization.Localization(__file__, 83, 23), amin_480196, *[cdist_call_result_480204], **kwargs_480207)
        
        # Processing the call keyword arguments (line 83)
        kwargs_480209 = {}
        # Getting the type of 'max' (line 83)
        max_480194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'max', False)
        # Calling max(args, kwargs) (line 83)
        max_call_result_480210 = invoke(stypy.reporting.localization.Localization(__file__, 83, 19), max_480194, *[amin_call_result_480208], **kwargs_480209)
        
        # Assigning a type to the variable 'expected' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'expected', max_call_result_480210)
        
        # Call to assert_almost_equal(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'actual' (line 85)
        actual_480212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'actual', False)
        # Getting the type of 'expected' (line 85)
        expected_480213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'expected', False)
        # Processing the call keyword arguments (line 85)
        int_480214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 54), 'int')
        keyword_480215 = int_480214
        kwargs_480216 = {'decimal': keyword_480215}
        # Getting the type of 'assert_almost_equal' (line 85)
        assert_almost_equal_480211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 85)
        assert_almost_equal_call_result_480217 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assert_almost_equal_480211, *[actual_480212, expected_480213], **kwargs_480216)
        
        
        # ################# End of 'test_4d_data_reverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_4d_data_reverse' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_480218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480218)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_4d_data_reverse'
        return stypy_return_type_480218


    @norecursion
    def test_indices(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_indices'
        module_type_store = module_type_store.open_function_context('test_indices', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_indices')
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_indices.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_indices', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_indices', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_indices(...)' code ##################

        
        # Assigning a Call to a Name (line 90):
        
        # Call to array(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_480221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_480222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_480223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 34), list_480222, int_480223)
        # Adding element type (line 90)
        int_480224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 34), list_480222, int_480224)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 33), list_480221, list_480222)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_480225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_480226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 43), list_480225, int_480226)
        # Adding element type (line 90)
        int_480227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 43), list_480225, int_480227)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 33), list_480221, list_480225)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_480228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_480229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 50), list_480228, int_480229)
        # Adding element type (line 90)
        int_480230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 50), list_480228, int_480230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 33), list_480221, list_480228)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_480231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_480232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 57), list_480231, int_480232)
        # Adding element type (line 90)
        int_480233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 57), list_480231, int_480233)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 33), list_480221, list_480231)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_480234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_480235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 64), list_480234, int_480235)
        # Adding element type (line 90)
        int_480236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 64), list_480234, int_480236)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 33), list_480221, list_480234)
        
        # Processing the call keyword arguments (line 90)
        kwargs_480237 = {}
        # Getting the type of 'np' (line 90)
        np_480219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'np', False)
        # Obtaining the member 'array' of a type (line 90)
        array_480220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 24), np_480219, 'array')
        # Calling array(args, kwargs) (line 90)
        array_call_result_480238 = invoke(stypy.reporting.localization.Localization(__file__, 90, 24), array_480220, *[list_480221], **kwargs_480237)
        
        # Assigning a type to the variable 'path_simple_1' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'path_simple_1', array_call_result_480238)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to array(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_480241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_480242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        int_480243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 34), list_480242, int_480243)
        # Adding element type (line 91)
        int_480244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 34), list_480242, int_480244)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 33), list_480241, list_480242)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_480245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        int_480246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 41), list_480245, int_480246)
        # Adding element type (line 91)
        int_480247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 41), list_480245, int_480247)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 33), list_480241, list_480245)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_480248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        int_480249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 48), list_480248, int_480249)
        # Adding element type (line 91)
        int_480250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 48), list_480248, int_480250)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 33), list_480241, list_480248)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_480251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        int_480252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 57), list_480251, int_480252)
        # Adding element type (line 91)
        int_480253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 57), list_480251, int_480253)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 33), list_480241, list_480251)
        
        # Processing the call keyword arguments (line 91)
        kwargs_480254 = {}
        # Getting the type of 'np' (line 91)
        np_480239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'np', False)
        # Obtaining the member 'array' of a type (line 91)
        array_480240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), np_480239, 'array')
        # Calling array(args, kwargs) (line 91)
        array_call_result_480255 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), array_480240, *[list_480241], **kwargs_480254)
        
        # Assigning a type to the variable 'path_simple_2' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'path_simple_2', array_call_result_480255)
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_480256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 66), 'int')
        slice_480257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 92, 17), int_480256, None, None)
        
        # Call to directed_hausdorff(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'path_simple_2' (line 92)
        path_simple_2_480259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'path_simple_2', False)
        # Getting the type of 'path_simple_1' (line 92)
        path_simple_1_480260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 51), 'path_simple_1', False)
        # Processing the call keyword arguments (line 92)
        kwargs_480261 = {}
        # Getting the type of 'directed_hausdorff' (line 92)
        directed_hausdorff_480258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 92)
        directed_hausdorff_call_result_480262 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), directed_hausdorff_480258, *[path_simple_2_480259, path_simple_1_480260], **kwargs_480261)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___480263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), directed_hausdorff_call_result_480262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_480264 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), getitem___480263, slice_480257)
        
        # Assigning a type to the variable 'actual' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'actual', subscript_call_result_480264)
        
        # Assigning a Tuple to a Name (line 93):
        
        # Obtaining an instance of the builtin type 'tuple' (line 93)
        tuple_480265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 93)
        # Adding element type (line 93)
        int_480266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 20), tuple_480265, int_480266)
        # Adding element type (line 93)
        int_480267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 20), tuple_480265, int_480267)
        
        # Assigning a type to the variable 'expected' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'expected', tuple_480265)
        
        # Call to assert_array_equal(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'actual' (line 94)
        actual_480269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'actual', False)
        # Getting the type of 'expected' (line 94)
        expected_480270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'expected', False)
        # Processing the call keyword arguments (line 94)
        kwargs_480271 = {}
        # Getting the type of 'assert_array_equal' (line 94)
        assert_array_equal_480268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 94)
        assert_array_equal_call_result_480272 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assert_array_equal_480268, *[actual_480269, expected_480270], **kwargs_480271)
        
        
        # ################# End of 'test_indices(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_indices' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_480273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480273)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_indices'
        return stypy_return_type_480273


    @norecursion
    def test_random_state(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_state'
        module_type_store = module_type_store.open_function_context('test_random_state', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_random_state')
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_random_state.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_random_state', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_state', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_state(...)' code ##################

        
        # Assigning a Call to a Name (line 99):
        
        # Call to check_random_state(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'None' (line 99)
        None_480275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 32), 'None', False)
        # Processing the call keyword arguments (line 99)
        kwargs_480276 = {}
        # Getting the type of 'check_random_state' (line 99)
        check_random_state_480274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'check_random_state', False)
        # Calling check_random_state(args, kwargs) (line 99)
        check_random_state_call_result_480277 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), check_random_state_480274, *[None_480275], **kwargs_480276)
        
        # Assigning a type to the variable 'rs' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'rs', check_random_state_call_result_480277)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to get_state(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_480280 = {}
        # Getting the type of 'rs' (line 100)
        rs_480278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'rs', False)
        # Obtaining the member 'get_state' of a type (line 100)
        get_state_480279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 27), rs_480278, 'get_state')
        # Calling get_state(args, kwargs) (line 100)
        get_state_call_result_480281 = invoke(stypy.reporting.localization.Localization(__file__, 100, 27), get_state_480279, *[], **kwargs_480280)
        
        # Assigning a type to the variable 'old_global_state' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'old_global_state', get_state_call_result_480281)
        
        # Call to directed_hausdorff(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'self' (line 101)
        self_480283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'self', False)
        # Obtaining the member 'path_1' of a type (line 101)
        path_1_480284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), self_480283, 'path_1')
        # Getting the type of 'self' (line 101)
        self_480285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'self', False)
        # Obtaining the member 'path_2' of a type (line 101)
        path_2_480286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 40), self_480285, 'path_2')
        # Processing the call keyword arguments (line 101)
        kwargs_480287 = {}
        # Getting the type of 'directed_hausdorff' (line 101)
        directed_hausdorff_480282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 101)
        directed_hausdorff_call_result_480288 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), directed_hausdorff_480282, *[path_1_480284, path_2_480286], **kwargs_480287)
        
        
        # Assigning a Call to a Name (line 102):
        
        # Call to check_random_state(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'None' (line 102)
        None_480290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 33), 'None', False)
        # Processing the call keyword arguments (line 102)
        kwargs_480291 = {}
        # Getting the type of 'check_random_state' (line 102)
        check_random_state_480289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'check_random_state', False)
        # Calling check_random_state(args, kwargs) (line 102)
        check_random_state_call_result_480292 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), check_random_state_480289, *[None_480290], **kwargs_480291)
        
        # Assigning a type to the variable 'rs2' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'rs2', check_random_state_call_result_480292)
        
        # Assigning a Call to a Name (line 103):
        
        # Call to get_state(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_480295 = {}
        # Getting the type of 'rs2' (line 103)
        rs2_480293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'rs2', False)
        # Obtaining the member 'get_state' of a type (line 103)
        get_state_480294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), rs2_480293, 'get_state')
        # Calling get_state(args, kwargs) (line 103)
        get_state_call_result_480296 = invoke(stypy.reporting.localization.Localization(__file__, 103, 27), get_state_480294, *[], **kwargs_480295)
        
        # Assigning a type to the variable 'new_global_state' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'new_global_state', get_state_call_result_480296)
        
        # Call to assert_equal(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'new_global_state' (line 104)
        new_global_state_480298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'new_global_state', False)
        # Getting the type of 'old_global_state' (line 104)
        old_global_state_480299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'old_global_state', False)
        # Processing the call keyword arguments (line 104)
        kwargs_480300 = {}
        # Getting the type of 'assert_equal' (line 104)
        assert_equal_480297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 104)
        assert_equal_call_result_480301 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assert_equal_480297, *[new_global_state_480298, old_global_state_480299], **kwargs_480300)
        
        
        # ################# End of 'test_random_state(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_state' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_480302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_state'
        return stypy_return_type_480302


    @norecursion
    def test_random_state_None_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_state_None_int'
        module_type_store = module_type_store.open_function_context('test_random_state_None_int', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_localization', localization)
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_function_name', 'TestHausdorff.test_random_state_None_int')
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_param_names_list', [])
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHausdorff.test_random_state_None_int.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.test_random_state_None_int', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_state_None_int', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_state_None_int(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_480303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        # Getting the type of 'None' (line 109)
        None_480304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_480303, None_480304)
        # Adding element type (line 109)
        int_480305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_480303, int_480305)
        
        # Testing the type of a for loop iterable (line 109)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 8), list_480303)
        # Getting the type of the for loop variable (line 109)
        for_loop_var_480306 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 8), list_480303)
        # Assigning a type to the variable 'seed' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'seed', for_loop_var_480306)
        # SSA begins for a for statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 110):
        
        # Call to check_random_state(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'None' (line 110)
        None_480308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 36), 'None', False)
        # Processing the call keyword arguments (line 110)
        kwargs_480309 = {}
        # Getting the type of 'check_random_state' (line 110)
        check_random_state_480307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'check_random_state', False)
        # Calling check_random_state(args, kwargs) (line 110)
        check_random_state_call_result_480310 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), check_random_state_480307, *[None_480308], **kwargs_480309)
        
        # Assigning a type to the variable 'rs' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'rs', check_random_state_call_result_480310)
        
        # Assigning a Call to a Name (line 111):
        
        # Call to get_state(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_480313 = {}
        # Getting the type of 'rs' (line 111)
        rs_480311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 31), 'rs', False)
        # Obtaining the member 'get_state' of a type (line 111)
        get_state_480312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 31), rs_480311, 'get_state')
        # Calling get_state(args, kwargs) (line 111)
        get_state_call_result_480314 = invoke(stypy.reporting.localization.Localization(__file__, 111, 31), get_state_480312, *[], **kwargs_480313)
        
        # Assigning a type to the variable 'old_global_state' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'old_global_state', get_state_call_result_480314)
        
        # Call to directed_hausdorff(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_480316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'self', False)
        # Obtaining the member 'path_1' of a type (line 112)
        path_1_480317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 31), self_480316, 'path_1')
        # Getting the type of 'self' (line 112)
        self_480318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 44), 'self', False)
        # Obtaining the member 'path_2' of a type (line 112)
        path_2_480319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 44), self_480318, 'path_2')
        # Getting the type of 'seed' (line 112)
        seed_480320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 57), 'seed', False)
        # Processing the call keyword arguments (line 112)
        kwargs_480321 = {}
        # Getting the type of 'directed_hausdorff' (line 112)
        directed_hausdorff_480315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'directed_hausdorff', False)
        # Calling directed_hausdorff(args, kwargs) (line 112)
        directed_hausdorff_call_result_480322 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), directed_hausdorff_480315, *[path_1_480317, path_2_480319, seed_480320], **kwargs_480321)
        
        
        # Assigning a Call to a Name (line 113):
        
        # Call to check_random_state(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'None' (line 113)
        None_480324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 37), 'None', False)
        # Processing the call keyword arguments (line 113)
        kwargs_480325 = {}
        # Getting the type of 'check_random_state' (line 113)
        check_random_state_480323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'check_random_state', False)
        # Calling check_random_state(args, kwargs) (line 113)
        check_random_state_call_result_480326 = invoke(stypy.reporting.localization.Localization(__file__, 113, 18), check_random_state_480323, *[None_480324], **kwargs_480325)
        
        # Assigning a type to the variable 'rs2' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'rs2', check_random_state_call_result_480326)
        
        # Assigning a Call to a Name (line 114):
        
        # Call to get_state(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_480329 = {}
        # Getting the type of 'rs2' (line 114)
        rs2_480327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 31), 'rs2', False)
        # Obtaining the member 'get_state' of a type (line 114)
        get_state_480328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 31), rs2_480327, 'get_state')
        # Calling get_state(args, kwargs) (line 114)
        get_state_call_result_480330 = invoke(stypy.reporting.localization.Localization(__file__, 114, 31), get_state_480328, *[], **kwargs_480329)
        
        # Assigning a type to the variable 'new_global_state' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'new_global_state', get_state_call_result_480330)
        
        # Call to assert_equal(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'new_global_state' (line 115)
        new_global_state_480332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'new_global_state', False)
        # Getting the type of 'old_global_state' (line 115)
        old_global_state_480333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 43), 'old_global_state', False)
        # Processing the call keyword arguments (line 115)
        kwargs_480334 = {}
        # Getting the type of 'assert_equal' (line 115)
        assert_equal_480331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 115)
        assert_equal_call_result_480335 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), assert_equal_480331, *[new_global_state_480332, old_global_state_480333], **kwargs_480334)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random_state_None_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_state_None_int' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_480336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_480336)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_state_None_int'
        return stypy_return_type_480336


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHausdorff.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHausdorff' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestHausdorff', TestHausdorff)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
