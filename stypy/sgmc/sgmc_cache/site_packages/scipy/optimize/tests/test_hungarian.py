
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Author: Brian M. Clapper, G. Varoquaux, Lars Buitinck
2: # License: BSD
3: 
4: from numpy.testing import assert_array_equal
5: from pytest import raises as assert_raises
6: 
7: import numpy as np
8: 
9: from scipy.optimize import linear_sum_assignment
10: 
11: 
12: def test_linear_sum_assignment():
13:     for cost_matrix, expected_cost in [
14:         # Square
15:         ([[400, 150, 400],
16:           [400, 450, 600],
17:           [300, 225, 300]],
18:          [150, 400, 300]
19:          ),
20: 
21:         # Rectangular variant
22:         ([[400, 150, 400, 1],
23:           [400, 450, 600, 2],
24:           [300, 225, 300, 3]],
25:          [150, 2, 300]),
26: 
27:         # Square
28:         ([[10, 10, 8],
29:           [9, 8, 1],
30:           [9, 7, 4]],
31:          [10, 1, 7]),
32: 
33:         # Rectangular variant
34:         ([[10, 10, 8, 11],
35:           [9, 8, 1, 1],
36:           [9, 7, 4, 10]],
37:          [10, 1, 4]),
38: 
39:         # n == 2, m == 0 matrix
40:         ([[], []],
41:          []),
42:     ]:
43:         cost_matrix = np.array(cost_matrix)
44:         row_ind, col_ind = linear_sum_assignment(cost_matrix)
45:         assert_array_equal(row_ind, np.sort(row_ind))
46:         assert_array_equal(expected_cost, cost_matrix[row_ind, col_ind])
47: 
48:         cost_matrix = cost_matrix.T
49:         row_ind, col_ind = linear_sum_assignment(cost_matrix)
50:         assert_array_equal(row_ind, np.sort(row_ind))
51:         assert_array_equal(np.sort(expected_cost),
52:                            np.sort(cost_matrix[row_ind, col_ind]))
53: 
54: 
55: def test_linear_sum_assignment_input_validation():
56:     assert_raises(ValueError, linear_sum_assignment, [1, 2, 3])
57: 
58:     C = [[1, 2, 3], [4, 5, 6]]
59:     assert_array_equal(linear_sum_assignment(C),
60:                        linear_sum_assignment(np.asarray(C)))
61:     assert_array_equal(linear_sum_assignment(C),
62:                        linear_sum_assignment(np.matrix(C)))
63: 
64:     I = np.identity(3)
65:     assert_array_equal(linear_sum_assignment(I.astype(np.bool)),
66:                        linear_sum_assignment(I))
67:     assert_raises(ValueError, linear_sum_assignment, I.astype(str))
68: 
69:     I[0][0] = np.nan
70:     assert_raises(ValueError, linear_sum_assignment, I)
71: 
72:     I = np.identity(3)
73:     I[1][1] = np.inf
74:     assert_raises(ValueError, linear_sum_assignment, I)
75: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205291 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_205291) is not StypyTypeError):

    if (import_205291 != 'pyd_module'):
        __import__(import_205291)
        sys_modules_205292 = sys.modules[import_205291]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_205292.module_type_store, module_type_store, ['assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_205292, sys_modules_205292.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal'], [assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_205291)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from pytest import assert_raises' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205293 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_205293) is not StypyTypeError):

    if (import_205293 != 'pyd_module'):
        __import__(import_205293)
        sys_modules_205294 = sys.modules[import_205293]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_205294.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_205294, sys_modules_205294.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_205293)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205295 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_205295) is not StypyTypeError):

    if (import_205295 != 'pyd_module'):
        __import__(import_205295)
        sys_modules_205296 = sys.modules[import_205295]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_205296.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_205295)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.optimize import linear_sum_assignment' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205297 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize')

if (type(import_205297) is not StypyTypeError):

    if (import_205297 != 'pyd_module'):
        __import__(import_205297)
        sys_modules_205298 = sys.modules[import_205297]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', sys_modules_205298.module_type_store, module_type_store, ['linear_sum_assignment'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_205298, sys_modules_205298.module_type_store, module_type_store)
    else:
        from scipy.optimize import linear_sum_assignment

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', None, module_type_store, ['linear_sum_assignment'], [linear_sum_assignment])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', import_205297)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def test_linear_sum_assignment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_linear_sum_assignment'
    module_type_store = module_type_store.open_function_context('test_linear_sum_assignment', 12, 0, False)
    
    # Passed parameters checking function
    test_linear_sum_assignment.stypy_localization = localization
    test_linear_sum_assignment.stypy_type_of_self = None
    test_linear_sum_assignment.stypy_type_store = module_type_store
    test_linear_sum_assignment.stypy_function_name = 'test_linear_sum_assignment'
    test_linear_sum_assignment.stypy_param_names_list = []
    test_linear_sum_assignment.stypy_varargs_param_name = None
    test_linear_sum_assignment.stypy_kwargs_param_name = None
    test_linear_sum_assignment.stypy_call_defaults = defaults
    test_linear_sum_assignment.stypy_call_varargs = varargs
    test_linear_sum_assignment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_linear_sum_assignment', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_linear_sum_assignment', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_linear_sum_assignment(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_205299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    
    # Obtaining an instance of the builtin type 'tuple' (line 15)
    tuple_205300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 15)
    # Adding element type (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_205301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_205302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_205303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_205302, int_205303)
    # Adding element type (line 15)
    int_205304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_205302, int_205304)
    # Adding element type (line 15)
    int_205305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_205302, int_205305)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), list_205301, list_205302)
    # Adding element type (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_205306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_205307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_205306, int_205307)
    # Adding element type (line 16)
    int_205308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_205306, int_205308)
    # Adding element type (line 16)
    int_205309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_205306, int_205309)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), list_205301, list_205306)
    # Adding element type (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_205310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_205311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_205310, int_205311)
    # Adding element type (line 17)
    int_205312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_205310, int_205312)
    # Adding element type (line 17)
    int_205313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_205310, int_205313)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), list_205301, list_205310)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), tuple_205300, list_205301)
    # Adding element type (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_205314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    int_205315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 9), list_205314, int_205315)
    # Adding element type (line 18)
    int_205316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 9), list_205314, int_205316)
    # Adding element type (line 18)
    int_205317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 9), list_205314, int_205317)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), tuple_205300, list_205314)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 38), list_205299, tuple_205300)
    # Adding element type (line 13)
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_205318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_205319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_205320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_205321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_205320, int_205321)
    # Adding element type (line 22)
    int_205322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_205320, int_205322)
    # Adding element type (line 22)
    int_205323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_205320, int_205323)
    # Adding element type (line 22)
    int_205324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_205320, int_205324)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), list_205319, list_205320)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_205325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    int_205326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 10), list_205325, int_205326)
    # Adding element type (line 23)
    int_205327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 10), list_205325, int_205327)
    # Adding element type (line 23)
    int_205328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 10), list_205325, int_205328)
    # Adding element type (line 23)
    int_205329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 10), list_205325, int_205329)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), list_205319, list_205325)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_205330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    int_205331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_205330, int_205331)
    # Adding element type (line 24)
    int_205332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_205330, int_205332)
    # Adding element type (line 24)
    int_205333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_205330, int_205333)
    # Adding element type (line 24)
    int_205334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_205330, int_205334)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), list_205319, list_205330)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_205318, list_205319)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_205335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_205336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), list_205335, int_205336)
    # Adding element type (line 25)
    int_205337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), list_205335, int_205337)
    # Adding element type (line 25)
    int_205338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), list_205335, int_205338)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_205318, list_205335)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 38), list_205299, tuple_205318)
    # Adding element type (line 13)
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_205339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_205340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_205341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_205342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_205341, int_205342)
    # Adding element type (line 28)
    int_205343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_205341, int_205343)
    # Adding element type (line 28)
    int_205344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_205341, int_205344)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), list_205340, list_205341)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_205345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    int_205346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_205345, int_205346)
    # Adding element type (line 29)
    int_205347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_205345, int_205347)
    # Adding element type (line 29)
    int_205348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_205345, int_205348)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), list_205340, list_205345)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_205349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    int_205350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), list_205349, int_205350)
    # Adding element type (line 30)
    int_205351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), list_205349, int_205351)
    # Adding element type (line 30)
    int_205352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), list_205349, int_205352)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), list_205340, list_205349)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_205339, list_205340)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_205353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    int_205354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), list_205353, int_205354)
    # Adding element type (line 31)
    int_205355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), list_205353, int_205355)
    # Adding element type (line 31)
    int_205356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), list_205353, int_205356)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_205339, list_205353)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 38), list_205299, tuple_205339)
    # Adding element type (line 13)
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_205357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_205358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_205359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    int_205360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 10), list_205359, int_205360)
    # Adding element type (line 34)
    int_205361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 10), list_205359, int_205361)
    # Adding element type (line 34)
    int_205362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 10), list_205359, int_205362)
    # Adding element type (line 34)
    int_205363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 10), list_205359, int_205363)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), list_205358, list_205359)
    # Adding element type (line 34)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_205364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    int_205365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_205364, int_205365)
    # Adding element type (line 35)
    int_205366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_205364, int_205366)
    # Adding element type (line 35)
    int_205367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_205364, int_205367)
    # Adding element type (line 35)
    int_205368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_205364, int_205368)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), list_205358, list_205364)
    # Adding element type (line 34)
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_205369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    int_205370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 10), list_205369, int_205370)
    # Adding element type (line 36)
    int_205371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 10), list_205369, int_205371)
    # Adding element type (line 36)
    int_205372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 10), list_205369, int_205372)
    # Adding element type (line 36)
    int_205373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 10), list_205369, int_205373)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), list_205358, list_205369)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), tuple_205357, list_205358)
    # Adding element type (line 34)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_205374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    int_205375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), list_205374, int_205375)
    # Adding element type (line 37)
    int_205376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), list_205374, int_205376)
    # Adding element type (line 37)
    int_205377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), list_205374, int_205377)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), tuple_205357, list_205374)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 38), list_205299, tuple_205357)
    # Adding element type (line 13)
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_205378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_205379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_205380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), list_205379, list_205380)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_205381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), list_205379, list_205381)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_205378, list_205379)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_205382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_205378, list_205382)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 38), list_205299, tuple_205378)
    
    # Testing the type of a for loop iterable (line 13)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 13, 4), list_205299)
    # Getting the type of the for loop variable (line 13)
    for_loop_var_205383 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 13, 4), list_205299)
    # Assigning a type to the variable 'cost_matrix' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'cost_matrix', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), for_loop_var_205383))
    # Assigning a type to the variable 'expected_cost' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'expected_cost', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), for_loop_var_205383))
    # SSA begins for a for statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to array(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'cost_matrix' (line 43)
    cost_matrix_205386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'cost_matrix', False)
    # Processing the call keyword arguments (line 43)
    kwargs_205387 = {}
    # Getting the type of 'np' (line 43)
    np_205384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'np', False)
    # Obtaining the member 'array' of a type (line 43)
    array_205385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 22), np_205384, 'array')
    # Calling array(args, kwargs) (line 43)
    array_call_result_205388 = invoke(stypy.reporting.localization.Localization(__file__, 43, 22), array_205385, *[cost_matrix_205386], **kwargs_205387)
    
    # Assigning a type to the variable 'cost_matrix' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'cost_matrix', array_call_result_205388)
    
    # Assigning a Call to a Tuple (line 44):
    
    # Assigning a Subscript to a Name (line 44):
    
    # Obtaining the type of the subscript
    int_205389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'int')
    
    # Call to linear_sum_assignment(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'cost_matrix' (line 44)
    cost_matrix_205391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 49), 'cost_matrix', False)
    # Processing the call keyword arguments (line 44)
    kwargs_205392 = {}
    # Getting the type of 'linear_sum_assignment' (line 44)
    linear_sum_assignment_205390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 44)
    linear_sum_assignment_call_result_205393 = invoke(stypy.reporting.localization.Localization(__file__, 44, 27), linear_sum_assignment_205390, *[cost_matrix_205391], **kwargs_205392)
    
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___205394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), linear_sum_assignment_call_result_205393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_205395 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), getitem___205394, int_205389)
    
    # Assigning a type to the variable 'tuple_var_assignment_205287' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_var_assignment_205287', subscript_call_result_205395)
    
    # Assigning a Subscript to a Name (line 44):
    
    # Obtaining the type of the subscript
    int_205396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'int')
    
    # Call to linear_sum_assignment(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'cost_matrix' (line 44)
    cost_matrix_205398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 49), 'cost_matrix', False)
    # Processing the call keyword arguments (line 44)
    kwargs_205399 = {}
    # Getting the type of 'linear_sum_assignment' (line 44)
    linear_sum_assignment_205397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 44)
    linear_sum_assignment_call_result_205400 = invoke(stypy.reporting.localization.Localization(__file__, 44, 27), linear_sum_assignment_205397, *[cost_matrix_205398], **kwargs_205399)
    
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___205401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), linear_sum_assignment_call_result_205400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_205402 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), getitem___205401, int_205396)
    
    # Assigning a type to the variable 'tuple_var_assignment_205288' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_var_assignment_205288', subscript_call_result_205402)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_var_assignment_205287' (line 44)
    tuple_var_assignment_205287_205403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_var_assignment_205287')
    # Assigning a type to the variable 'row_ind' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'row_ind', tuple_var_assignment_205287_205403)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_var_assignment_205288' (line 44)
    tuple_var_assignment_205288_205404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_var_assignment_205288')
    # Assigning a type to the variable 'col_ind' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'col_ind', tuple_var_assignment_205288_205404)
    
    # Call to assert_array_equal(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'row_ind' (line 45)
    row_ind_205406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'row_ind', False)
    
    # Call to sort(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'row_ind' (line 45)
    row_ind_205409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'row_ind', False)
    # Processing the call keyword arguments (line 45)
    kwargs_205410 = {}
    # Getting the type of 'np' (line 45)
    np_205407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'np', False)
    # Obtaining the member 'sort' of a type (line 45)
    sort_205408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 36), np_205407, 'sort')
    # Calling sort(args, kwargs) (line 45)
    sort_call_result_205411 = invoke(stypy.reporting.localization.Localization(__file__, 45, 36), sort_205408, *[row_ind_205409], **kwargs_205410)
    
    # Processing the call keyword arguments (line 45)
    kwargs_205412 = {}
    # Getting the type of 'assert_array_equal' (line 45)
    assert_array_equal_205405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 45)
    assert_array_equal_call_result_205413 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assert_array_equal_205405, *[row_ind_205406, sort_call_result_205411], **kwargs_205412)
    
    
    # Call to assert_array_equal(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'expected_cost' (line 46)
    expected_cost_205415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'expected_cost', False)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_205416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'row_ind' (line 46)
    row_ind_205417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 'row_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 54), tuple_205416, row_ind_205417)
    # Adding element type (line 46)
    # Getting the type of 'col_ind' (line 46)
    col_ind_205418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 63), 'col_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 54), tuple_205416, col_ind_205418)
    
    # Getting the type of 'cost_matrix' (line 46)
    cost_matrix_205419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'cost_matrix', False)
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___205420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 42), cost_matrix_205419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_205421 = invoke(stypy.reporting.localization.Localization(__file__, 46, 42), getitem___205420, tuple_205416)
    
    # Processing the call keyword arguments (line 46)
    kwargs_205422 = {}
    # Getting the type of 'assert_array_equal' (line 46)
    assert_array_equal_205414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 46)
    assert_array_equal_call_result_205423 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assert_array_equal_205414, *[expected_cost_205415, subscript_call_result_205421], **kwargs_205422)
    
    
    # Assigning a Attribute to a Name (line 48):
    
    # Assigning a Attribute to a Name (line 48):
    # Getting the type of 'cost_matrix' (line 48)
    cost_matrix_205424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'cost_matrix')
    # Obtaining the member 'T' of a type (line 48)
    T_205425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), cost_matrix_205424, 'T')
    # Assigning a type to the variable 'cost_matrix' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'cost_matrix', T_205425)
    
    # Assigning a Call to a Tuple (line 49):
    
    # Assigning a Subscript to a Name (line 49):
    
    # Obtaining the type of the subscript
    int_205426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
    
    # Call to linear_sum_assignment(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'cost_matrix' (line 49)
    cost_matrix_205428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'cost_matrix', False)
    # Processing the call keyword arguments (line 49)
    kwargs_205429 = {}
    # Getting the type of 'linear_sum_assignment' (line 49)
    linear_sum_assignment_205427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 27), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 49)
    linear_sum_assignment_call_result_205430 = invoke(stypy.reporting.localization.Localization(__file__, 49, 27), linear_sum_assignment_205427, *[cost_matrix_205428], **kwargs_205429)
    
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___205431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), linear_sum_assignment_call_result_205430, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_205432 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___205431, int_205426)
    
    # Assigning a type to the variable 'tuple_var_assignment_205289' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_205289', subscript_call_result_205432)
    
    # Assigning a Subscript to a Name (line 49):
    
    # Obtaining the type of the subscript
    int_205433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
    
    # Call to linear_sum_assignment(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'cost_matrix' (line 49)
    cost_matrix_205435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'cost_matrix', False)
    # Processing the call keyword arguments (line 49)
    kwargs_205436 = {}
    # Getting the type of 'linear_sum_assignment' (line 49)
    linear_sum_assignment_205434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 27), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 49)
    linear_sum_assignment_call_result_205437 = invoke(stypy.reporting.localization.Localization(__file__, 49, 27), linear_sum_assignment_205434, *[cost_matrix_205435], **kwargs_205436)
    
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___205438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), linear_sum_assignment_call_result_205437, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_205439 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___205438, int_205433)
    
    # Assigning a type to the variable 'tuple_var_assignment_205290' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_205290', subscript_call_result_205439)
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'tuple_var_assignment_205289' (line 49)
    tuple_var_assignment_205289_205440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_205289')
    # Assigning a type to the variable 'row_ind' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'row_ind', tuple_var_assignment_205289_205440)
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'tuple_var_assignment_205290' (line 49)
    tuple_var_assignment_205290_205441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_205290')
    # Assigning a type to the variable 'col_ind' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'col_ind', tuple_var_assignment_205290_205441)
    
    # Call to assert_array_equal(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'row_ind' (line 50)
    row_ind_205443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'row_ind', False)
    
    # Call to sort(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'row_ind' (line 50)
    row_ind_205446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 44), 'row_ind', False)
    # Processing the call keyword arguments (line 50)
    kwargs_205447 = {}
    # Getting the type of 'np' (line 50)
    np_205444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'np', False)
    # Obtaining the member 'sort' of a type (line 50)
    sort_205445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 36), np_205444, 'sort')
    # Calling sort(args, kwargs) (line 50)
    sort_call_result_205448 = invoke(stypy.reporting.localization.Localization(__file__, 50, 36), sort_205445, *[row_ind_205446], **kwargs_205447)
    
    # Processing the call keyword arguments (line 50)
    kwargs_205449 = {}
    # Getting the type of 'assert_array_equal' (line 50)
    assert_array_equal_205442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 50)
    assert_array_equal_call_result_205450 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert_array_equal_205442, *[row_ind_205443, sort_call_result_205448], **kwargs_205449)
    
    
    # Call to assert_array_equal(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to sort(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'expected_cost' (line 51)
    expected_cost_205454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 35), 'expected_cost', False)
    # Processing the call keyword arguments (line 51)
    kwargs_205455 = {}
    # Getting the type of 'np' (line 51)
    np_205452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 27), 'np', False)
    # Obtaining the member 'sort' of a type (line 51)
    sort_205453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 27), np_205452, 'sort')
    # Calling sort(args, kwargs) (line 51)
    sort_call_result_205456 = invoke(stypy.reporting.localization.Localization(__file__, 51, 27), sort_205453, *[expected_cost_205454], **kwargs_205455)
    
    
    # Call to sort(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 52)
    tuple_205459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 52)
    # Adding element type (line 52)
    # Getting the type of 'row_ind' (line 52)
    row_ind_205460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 47), 'row_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 47), tuple_205459, row_ind_205460)
    # Adding element type (line 52)
    # Getting the type of 'col_ind' (line 52)
    col_ind_205461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 56), 'col_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 47), tuple_205459, col_ind_205461)
    
    # Getting the type of 'cost_matrix' (line 52)
    cost_matrix_205462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'cost_matrix', False)
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___205463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 35), cost_matrix_205462, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_205464 = invoke(stypy.reporting.localization.Localization(__file__, 52, 35), getitem___205463, tuple_205459)
    
    # Processing the call keyword arguments (line 52)
    kwargs_205465 = {}
    # Getting the type of 'np' (line 52)
    np_205457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 27), 'np', False)
    # Obtaining the member 'sort' of a type (line 52)
    sort_205458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 27), np_205457, 'sort')
    # Calling sort(args, kwargs) (line 52)
    sort_call_result_205466 = invoke(stypy.reporting.localization.Localization(__file__, 52, 27), sort_205458, *[subscript_call_result_205464], **kwargs_205465)
    
    # Processing the call keyword arguments (line 51)
    kwargs_205467 = {}
    # Getting the type of 'assert_array_equal' (line 51)
    assert_array_equal_205451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 51)
    assert_array_equal_call_result_205468 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert_array_equal_205451, *[sort_call_result_205456, sort_call_result_205466], **kwargs_205467)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_linear_sum_assignment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_linear_sum_assignment' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_205469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205469)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_linear_sum_assignment'
    return stypy_return_type_205469

# Assigning a type to the variable 'test_linear_sum_assignment' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'test_linear_sum_assignment', test_linear_sum_assignment)

@norecursion
def test_linear_sum_assignment_input_validation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_linear_sum_assignment_input_validation'
    module_type_store = module_type_store.open_function_context('test_linear_sum_assignment_input_validation', 55, 0, False)
    
    # Passed parameters checking function
    test_linear_sum_assignment_input_validation.stypy_localization = localization
    test_linear_sum_assignment_input_validation.stypy_type_of_self = None
    test_linear_sum_assignment_input_validation.stypy_type_store = module_type_store
    test_linear_sum_assignment_input_validation.stypy_function_name = 'test_linear_sum_assignment_input_validation'
    test_linear_sum_assignment_input_validation.stypy_param_names_list = []
    test_linear_sum_assignment_input_validation.stypy_varargs_param_name = None
    test_linear_sum_assignment_input_validation.stypy_kwargs_param_name = None
    test_linear_sum_assignment_input_validation.stypy_call_defaults = defaults
    test_linear_sum_assignment_input_validation.stypy_call_varargs = varargs
    test_linear_sum_assignment_input_validation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_linear_sum_assignment_input_validation', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_linear_sum_assignment_input_validation', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_linear_sum_assignment_input_validation(...)' code ##################

    
    # Call to assert_raises(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'ValueError' (line 56)
    ValueError_205471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'ValueError', False)
    # Getting the type of 'linear_sum_assignment' (line 56)
    linear_sum_assignment_205472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'linear_sum_assignment', False)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_205473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_205474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 53), list_205473, int_205474)
    # Adding element type (line 56)
    int_205475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 53), list_205473, int_205475)
    # Adding element type (line 56)
    int_205476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 53), list_205473, int_205476)
    
    # Processing the call keyword arguments (line 56)
    kwargs_205477 = {}
    # Getting the type of 'assert_raises' (line 56)
    assert_raises_205470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 56)
    assert_raises_call_result_205478 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), assert_raises_205470, *[ValueError_205471, linear_sum_assignment_205472, list_205473], **kwargs_205477)
    
    
    # Assigning a List to a Name (line 58):
    
    # Assigning a List to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_205479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_205480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    int_205481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), list_205480, int_205481)
    # Adding element type (line 58)
    int_205482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), list_205480, int_205482)
    # Adding element type (line 58)
    int_205483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), list_205480, int_205483)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), list_205479, list_205480)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_205484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    int_205485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 20), list_205484, int_205485)
    # Adding element type (line 58)
    int_205486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 20), list_205484, int_205486)
    # Adding element type (line 58)
    int_205487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 20), list_205484, int_205487)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), list_205479, list_205484)
    
    # Assigning a type to the variable 'C' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'C', list_205479)
    
    # Call to assert_array_equal(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Call to linear_sum_assignment(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'C' (line 59)
    C_205490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'C', False)
    # Processing the call keyword arguments (line 59)
    kwargs_205491 = {}
    # Getting the type of 'linear_sum_assignment' (line 59)
    linear_sum_assignment_205489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 59)
    linear_sum_assignment_call_result_205492 = invoke(stypy.reporting.localization.Localization(__file__, 59, 23), linear_sum_assignment_205489, *[C_205490], **kwargs_205491)
    
    
    # Call to linear_sum_assignment(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Call to asarray(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'C' (line 60)
    C_205496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'C', False)
    # Processing the call keyword arguments (line 60)
    kwargs_205497 = {}
    # Getting the type of 'np' (line 60)
    np_205494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'np', False)
    # Obtaining the member 'asarray' of a type (line 60)
    asarray_205495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 45), np_205494, 'asarray')
    # Calling asarray(args, kwargs) (line 60)
    asarray_call_result_205498 = invoke(stypy.reporting.localization.Localization(__file__, 60, 45), asarray_205495, *[C_205496], **kwargs_205497)
    
    # Processing the call keyword arguments (line 60)
    kwargs_205499 = {}
    # Getting the type of 'linear_sum_assignment' (line 60)
    linear_sum_assignment_205493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 60)
    linear_sum_assignment_call_result_205500 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), linear_sum_assignment_205493, *[asarray_call_result_205498], **kwargs_205499)
    
    # Processing the call keyword arguments (line 59)
    kwargs_205501 = {}
    # Getting the type of 'assert_array_equal' (line 59)
    assert_array_equal_205488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 59)
    assert_array_equal_call_result_205502 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), assert_array_equal_205488, *[linear_sum_assignment_call_result_205492, linear_sum_assignment_call_result_205500], **kwargs_205501)
    
    
    # Call to assert_array_equal(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to linear_sum_assignment(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'C' (line 61)
    C_205505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'C', False)
    # Processing the call keyword arguments (line 61)
    kwargs_205506 = {}
    # Getting the type of 'linear_sum_assignment' (line 61)
    linear_sum_assignment_205504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 61)
    linear_sum_assignment_call_result_205507 = invoke(stypy.reporting.localization.Localization(__file__, 61, 23), linear_sum_assignment_205504, *[C_205505], **kwargs_205506)
    
    
    # Call to linear_sum_assignment(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Call to matrix(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'C' (line 62)
    C_205511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 55), 'C', False)
    # Processing the call keyword arguments (line 62)
    kwargs_205512 = {}
    # Getting the type of 'np' (line 62)
    np_205509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 45), 'np', False)
    # Obtaining the member 'matrix' of a type (line 62)
    matrix_205510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 45), np_205509, 'matrix')
    # Calling matrix(args, kwargs) (line 62)
    matrix_call_result_205513 = invoke(stypy.reporting.localization.Localization(__file__, 62, 45), matrix_205510, *[C_205511], **kwargs_205512)
    
    # Processing the call keyword arguments (line 62)
    kwargs_205514 = {}
    # Getting the type of 'linear_sum_assignment' (line 62)
    linear_sum_assignment_205508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 62)
    linear_sum_assignment_call_result_205515 = invoke(stypy.reporting.localization.Localization(__file__, 62, 23), linear_sum_assignment_205508, *[matrix_call_result_205513], **kwargs_205514)
    
    # Processing the call keyword arguments (line 61)
    kwargs_205516 = {}
    # Getting the type of 'assert_array_equal' (line 61)
    assert_array_equal_205503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 61)
    assert_array_equal_call_result_205517 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), assert_array_equal_205503, *[linear_sum_assignment_call_result_205507, linear_sum_assignment_call_result_205515], **kwargs_205516)
    
    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to identity(...): (line 64)
    # Processing the call arguments (line 64)
    int_205520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'int')
    # Processing the call keyword arguments (line 64)
    kwargs_205521 = {}
    # Getting the type of 'np' (line 64)
    np_205518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'np', False)
    # Obtaining the member 'identity' of a type (line 64)
    identity_205519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), np_205518, 'identity')
    # Calling identity(args, kwargs) (line 64)
    identity_call_result_205522 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), identity_205519, *[int_205520], **kwargs_205521)
    
    # Assigning a type to the variable 'I' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'I', identity_call_result_205522)
    
    # Call to assert_array_equal(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Call to linear_sum_assignment(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Call to astype(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'np' (line 65)
    np_205527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 54), 'np', False)
    # Obtaining the member 'bool' of a type (line 65)
    bool_205528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 54), np_205527, 'bool')
    # Processing the call keyword arguments (line 65)
    kwargs_205529 = {}
    # Getting the type of 'I' (line 65)
    I_205525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 45), 'I', False)
    # Obtaining the member 'astype' of a type (line 65)
    astype_205526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 45), I_205525, 'astype')
    # Calling astype(args, kwargs) (line 65)
    astype_call_result_205530 = invoke(stypy.reporting.localization.Localization(__file__, 65, 45), astype_205526, *[bool_205528], **kwargs_205529)
    
    # Processing the call keyword arguments (line 65)
    kwargs_205531 = {}
    # Getting the type of 'linear_sum_assignment' (line 65)
    linear_sum_assignment_205524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 65)
    linear_sum_assignment_call_result_205532 = invoke(stypy.reporting.localization.Localization(__file__, 65, 23), linear_sum_assignment_205524, *[astype_call_result_205530], **kwargs_205531)
    
    
    # Call to linear_sum_assignment(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'I' (line 66)
    I_205534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'I', False)
    # Processing the call keyword arguments (line 66)
    kwargs_205535 = {}
    # Getting the type of 'linear_sum_assignment' (line 66)
    linear_sum_assignment_205533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'linear_sum_assignment', False)
    # Calling linear_sum_assignment(args, kwargs) (line 66)
    linear_sum_assignment_call_result_205536 = invoke(stypy.reporting.localization.Localization(__file__, 66, 23), linear_sum_assignment_205533, *[I_205534], **kwargs_205535)
    
    # Processing the call keyword arguments (line 65)
    kwargs_205537 = {}
    # Getting the type of 'assert_array_equal' (line 65)
    assert_array_equal_205523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 65)
    assert_array_equal_call_result_205538 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), assert_array_equal_205523, *[linear_sum_assignment_call_result_205532, linear_sum_assignment_call_result_205536], **kwargs_205537)
    
    
    # Call to assert_raises(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'ValueError' (line 67)
    ValueError_205540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'ValueError', False)
    # Getting the type of 'linear_sum_assignment' (line 67)
    linear_sum_assignment_205541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'linear_sum_assignment', False)
    
    # Call to astype(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'str' (line 67)
    str_205544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 62), 'str', False)
    # Processing the call keyword arguments (line 67)
    kwargs_205545 = {}
    # Getting the type of 'I' (line 67)
    I_205542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 53), 'I', False)
    # Obtaining the member 'astype' of a type (line 67)
    astype_205543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 53), I_205542, 'astype')
    # Calling astype(args, kwargs) (line 67)
    astype_call_result_205546 = invoke(stypy.reporting.localization.Localization(__file__, 67, 53), astype_205543, *[str_205544], **kwargs_205545)
    
    # Processing the call keyword arguments (line 67)
    kwargs_205547 = {}
    # Getting the type of 'assert_raises' (line 67)
    assert_raises_205539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 67)
    assert_raises_call_result_205548 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), assert_raises_205539, *[ValueError_205540, linear_sum_assignment_205541, astype_call_result_205546], **kwargs_205547)
    
    
    # Assigning a Attribute to a Subscript (line 69):
    
    # Assigning a Attribute to a Subscript (line 69):
    # Getting the type of 'np' (line 69)
    np_205549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'np')
    # Obtaining the member 'nan' of a type (line 69)
    nan_205550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 14), np_205549, 'nan')
    
    # Obtaining the type of the subscript
    int_205551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 6), 'int')
    # Getting the type of 'I' (line 69)
    I_205552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'I')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___205553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), I_205552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_205554 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), getitem___205553, int_205551)
    
    int_205555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 9), 'int')
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 4), subscript_call_result_205554, (int_205555, nan_205550))
    
    # Call to assert_raises(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'ValueError' (line 70)
    ValueError_205557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'ValueError', False)
    # Getting the type of 'linear_sum_assignment' (line 70)
    linear_sum_assignment_205558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 30), 'linear_sum_assignment', False)
    # Getting the type of 'I' (line 70)
    I_205559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 53), 'I', False)
    # Processing the call keyword arguments (line 70)
    kwargs_205560 = {}
    # Getting the type of 'assert_raises' (line 70)
    assert_raises_205556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 70)
    assert_raises_call_result_205561 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), assert_raises_205556, *[ValueError_205557, linear_sum_assignment_205558, I_205559], **kwargs_205560)
    
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to identity(...): (line 72)
    # Processing the call arguments (line 72)
    int_205564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'int')
    # Processing the call keyword arguments (line 72)
    kwargs_205565 = {}
    # Getting the type of 'np' (line 72)
    np_205562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'np', False)
    # Obtaining the member 'identity' of a type (line 72)
    identity_205563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), np_205562, 'identity')
    # Calling identity(args, kwargs) (line 72)
    identity_call_result_205566 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), identity_205563, *[int_205564], **kwargs_205565)
    
    # Assigning a type to the variable 'I' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'I', identity_call_result_205566)
    
    # Assigning a Attribute to a Subscript (line 73):
    
    # Assigning a Attribute to a Subscript (line 73):
    # Getting the type of 'np' (line 73)
    np_205567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'np')
    # Obtaining the member 'inf' of a type (line 73)
    inf_205568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 14), np_205567, 'inf')
    
    # Obtaining the type of the subscript
    int_205569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 6), 'int')
    # Getting the type of 'I' (line 73)
    I_205570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'I')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___205571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 4), I_205570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_205572 = invoke(stypy.reporting.localization.Localization(__file__, 73, 4), getitem___205571, int_205569)
    
    int_205573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 9), 'int')
    # Storing an element on a container (line 73)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 4), subscript_call_result_205572, (int_205573, inf_205568))
    
    # Call to assert_raises(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'ValueError' (line 74)
    ValueError_205575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'ValueError', False)
    # Getting the type of 'linear_sum_assignment' (line 74)
    linear_sum_assignment_205576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), 'linear_sum_assignment', False)
    # Getting the type of 'I' (line 74)
    I_205577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 53), 'I', False)
    # Processing the call keyword arguments (line 74)
    kwargs_205578 = {}
    # Getting the type of 'assert_raises' (line 74)
    assert_raises_205574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 74)
    assert_raises_call_result_205579 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), assert_raises_205574, *[ValueError_205575, linear_sum_assignment_205576, I_205577], **kwargs_205578)
    
    
    # ################# End of 'test_linear_sum_assignment_input_validation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_linear_sum_assignment_input_validation' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_205580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205580)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_linear_sum_assignment_input_validation'
    return stypy_return_type_205580

# Assigning a type to the variable 'test_linear_sum_assignment_input_validation' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'test_linear_sum_assignment_input_validation', test_linear_sum_assignment_input_validation)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
