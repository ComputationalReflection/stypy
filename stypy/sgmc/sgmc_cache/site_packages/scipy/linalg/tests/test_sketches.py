
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for _sketches.py.'''
2: 
3: from __future__ import division, print_function, absolute_import
4: import numpy as np
5: from scipy.linalg import clarkson_woodruff_transform
6: 
7: from numpy.testing import assert_
8: 
9: 
10: def make_random_dense_gaussian_matrix(n_rows, n_columns, mu=0, sigma=0.01):
11:     '''
12:     Make some random data with Gaussian distributed values
13:     '''
14:     np.random.seed(142352345)
15:     res = np.random.normal(mu, sigma, n_rows*n_columns)
16:     return np.reshape(res, (n_rows, n_columns))
17: 
18: 
19: class TestClarksonWoodruffTransform(object):
20:     '''
21:     Testing the Clarkson Woodruff Transform
22:     '''
23:     # Big dense matrix dimensions
24:     n_matrix_rows = 2000
25:     n_matrix_columns = 100
26: 
27:     # Sketch matrix dimensions
28:     n_sketch_rows = 100
29: 
30:     # Error threshold
31:     threshold = 0.1
32: 
33:     dense_big_matrix = make_random_dense_gaussian_matrix(n_matrix_rows,
34:                                                          n_matrix_columns)
35: 
36:     def test_sketch_dimensions(self):
37:         sketch = clarkson_woodruff_transform(self.dense_big_matrix,
38:                                              self.n_sketch_rows)
39: 
40:         assert_(sketch.shape == (self.n_sketch_rows,
41:                                  self.dense_big_matrix.shape[1]))
42: 
43:     def test_sketch_rows_norm(self):
44:         # Given the probabilistic nature of the sketches
45:         # we run the 'test' multiple times and check that
46:         # we pass all/almost all the tries
47:         n_errors = 0
48: 
49:         seeds = [1755490010, 934377150, 1391612830, 1752708722, 2008891431,
50:                  1302443994, 1521083269, 1501189312, 1126232505, 1533465685]
51: 
52:         for seed_ in seeds:
53:             sketch = clarkson_woodruff_transform(self.dense_big_matrix,
54:                                                  self.n_sketch_rows, seed_)
55: 
56:             # We could use other norms (like L2)
57:             err = np.linalg.norm(self.dense_big_matrix) - np.linalg.norm(sketch)
58:             if err > self.threshold:
59:                 n_errors += 1
60: 
61:         assert_(n_errors == 0)
62: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_105641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for _sketches.py.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_105642 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_105642) is not StypyTypeError):

    if (import_105642 != 'pyd_module'):
        __import__(import_105642)
        sys_modules_105643 = sys.modules[import_105642]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_105643.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_105642)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.linalg import clarkson_woodruff_transform' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_105644 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg')

if (type(import_105644) is not StypyTypeError):

    if (import_105644 != 'pyd_module'):
        __import__(import_105644)
        sys_modules_105645 = sys.modules[import_105644]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', sys_modules_105645.module_type_store, module_type_store, ['clarkson_woodruff_transform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_105645, sys_modules_105645.module_type_store, module_type_store)
    else:
        from scipy.linalg import clarkson_woodruff_transform

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', None, module_type_store, ['clarkson_woodruff_transform'], [clarkson_woodruff_transform])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', import_105644)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_105646 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_105646) is not StypyTypeError):

    if (import_105646 != 'pyd_module'):
        __import__(import_105646)
        sys_modules_105647 = sys.modules[import_105646]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_105647.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_105647, sys_modules_105647.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_105646)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')


@norecursion
def make_random_dense_gaussian_matrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_105648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 60), 'int')
    float_105649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 69), 'float')
    defaults = [int_105648, float_105649]
    # Create a new context for function 'make_random_dense_gaussian_matrix'
    module_type_store = module_type_store.open_function_context('make_random_dense_gaussian_matrix', 10, 0, False)
    
    # Passed parameters checking function
    make_random_dense_gaussian_matrix.stypy_localization = localization
    make_random_dense_gaussian_matrix.stypy_type_of_self = None
    make_random_dense_gaussian_matrix.stypy_type_store = module_type_store
    make_random_dense_gaussian_matrix.stypy_function_name = 'make_random_dense_gaussian_matrix'
    make_random_dense_gaussian_matrix.stypy_param_names_list = ['n_rows', 'n_columns', 'mu', 'sigma']
    make_random_dense_gaussian_matrix.stypy_varargs_param_name = None
    make_random_dense_gaussian_matrix.stypy_kwargs_param_name = None
    make_random_dense_gaussian_matrix.stypy_call_defaults = defaults
    make_random_dense_gaussian_matrix.stypy_call_varargs = varargs
    make_random_dense_gaussian_matrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_random_dense_gaussian_matrix', ['n_rows', 'n_columns', 'mu', 'sigma'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_random_dense_gaussian_matrix', localization, ['n_rows', 'n_columns', 'mu', 'sigma'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_random_dense_gaussian_matrix(...)' code ##################

    str_105650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\n    Make some random data with Gaussian distributed values\n    ')
    
    # Call to seed(...): (line 14)
    # Processing the call arguments (line 14)
    int_105654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_105655 = {}
    # Getting the type of 'np' (line 14)
    np_105651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 14)
    random_105652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), np_105651, 'random')
    # Obtaining the member 'seed' of a type (line 14)
    seed_105653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), random_105652, 'seed')
    # Calling seed(args, kwargs) (line 14)
    seed_call_result_105656 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), seed_105653, *[int_105654], **kwargs_105655)
    
    
    # Assigning a Call to a Name (line 15):
    
    # Call to normal(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'mu' (line 15)
    mu_105660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'mu', False)
    # Getting the type of 'sigma' (line 15)
    sigma_105661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 31), 'sigma', False)
    # Getting the type of 'n_rows' (line 15)
    n_rows_105662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 38), 'n_rows', False)
    # Getting the type of 'n_columns' (line 15)
    n_columns_105663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 45), 'n_columns', False)
    # Applying the binary operator '*' (line 15)
    result_mul_105664 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 38), '*', n_rows_105662, n_columns_105663)
    
    # Processing the call keyword arguments (line 15)
    kwargs_105665 = {}
    # Getting the type of 'np' (line 15)
    np_105657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'np', False)
    # Obtaining the member 'random' of a type (line 15)
    random_105658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), np_105657, 'random')
    # Obtaining the member 'normal' of a type (line 15)
    normal_105659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), random_105658, 'normal')
    # Calling normal(args, kwargs) (line 15)
    normal_call_result_105666 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), normal_105659, *[mu_105660, sigma_105661, result_mul_105664], **kwargs_105665)
    
    # Assigning a type to the variable 'res' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'res', normal_call_result_105666)
    
    # Call to reshape(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'res' (line 16)
    res_105669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'res', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_105670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    # Getting the type of 'n_rows' (line 16)
    n_rows_105671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'n_rows', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 28), tuple_105670, n_rows_105671)
    # Adding element type (line 16)
    # Getting the type of 'n_columns' (line 16)
    n_columns_105672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 36), 'n_columns', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 28), tuple_105670, n_columns_105672)
    
    # Processing the call keyword arguments (line 16)
    kwargs_105673 = {}
    # Getting the type of 'np' (line 16)
    np_105667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'np', False)
    # Obtaining the member 'reshape' of a type (line 16)
    reshape_105668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), np_105667, 'reshape')
    # Calling reshape(args, kwargs) (line 16)
    reshape_call_result_105674 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), reshape_105668, *[res_105669, tuple_105670], **kwargs_105673)
    
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', reshape_call_result_105674)
    
    # ################# End of 'make_random_dense_gaussian_matrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_random_dense_gaussian_matrix' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_105675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_random_dense_gaussian_matrix'
    return stypy_return_type_105675

# Assigning a type to the variable 'make_random_dense_gaussian_matrix' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'make_random_dense_gaussian_matrix', make_random_dense_gaussian_matrix)
# Declaration of the 'TestClarksonWoodruffTransform' class

class TestClarksonWoodruffTransform(object, ):
    str_105676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n    Testing the Clarkson Woodruff Transform\n    ')

    @norecursion
    def test_sketch_dimensions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sketch_dimensions'
        module_type_store = module_type_store.open_function_context('test_sketch_dimensions', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_localization', localization)
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_function_name', 'TestClarksonWoodruffTransform.test_sketch_dimensions')
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_param_names_list', [])
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestClarksonWoodruffTransform.test_sketch_dimensions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestClarksonWoodruffTransform.test_sketch_dimensions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sketch_dimensions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sketch_dimensions(...)' code ##################

        
        # Assigning a Call to a Name (line 37):
        
        # Call to clarkson_woodruff_transform(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'self' (line 37)
        self_105678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 45), 'self', False)
        # Obtaining the member 'dense_big_matrix' of a type (line 37)
        dense_big_matrix_105679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 45), self_105678, 'dense_big_matrix')
        # Getting the type of 'self' (line 38)
        self_105680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 45), 'self', False)
        # Obtaining the member 'n_sketch_rows' of a type (line 38)
        n_sketch_rows_105681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 45), self_105680, 'n_sketch_rows')
        # Processing the call keyword arguments (line 37)
        kwargs_105682 = {}
        # Getting the type of 'clarkson_woodruff_transform' (line 37)
        clarkson_woodruff_transform_105677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'clarkson_woodruff_transform', False)
        # Calling clarkson_woodruff_transform(args, kwargs) (line 37)
        clarkson_woodruff_transform_call_result_105683 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), clarkson_woodruff_transform_105677, *[dense_big_matrix_105679, n_sketch_rows_105681], **kwargs_105682)
        
        # Assigning a type to the variable 'sketch' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'sketch', clarkson_woodruff_transform_call_result_105683)
        
        # Call to assert_(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Getting the type of 'sketch' (line 40)
        sketch_105685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'sketch', False)
        # Obtaining the member 'shape' of a type (line 40)
        shape_105686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), sketch_105685, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_105687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        # Getting the type of 'self' (line 40)
        self_105688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'self', False)
        # Obtaining the member 'n_sketch_rows' of a type (line 40)
        n_sketch_rows_105689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 33), self_105688, 'n_sketch_rows')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 33), tuple_105687, n_sketch_rows_105689)
        # Adding element type (line 40)
        
        # Obtaining the type of the subscript
        int_105690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 61), 'int')
        # Getting the type of 'self' (line 41)
        self_105691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 33), 'self', False)
        # Obtaining the member 'dense_big_matrix' of a type (line 41)
        dense_big_matrix_105692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 33), self_105691, 'dense_big_matrix')
        # Obtaining the member 'shape' of a type (line 41)
        shape_105693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 33), dense_big_matrix_105692, 'shape')
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___105694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 33), shape_105693, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_105695 = invoke(stypy.reporting.localization.Localization(__file__, 41, 33), getitem___105694, int_105690)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 33), tuple_105687, subscript_call_result_105695)
        
        # Applying the binary operator '==' (line 40)
        result_eq_105696 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), '==', shape_105686, tuple_105687)
        
        # Processing the call keyword arguments (line 40)
        kwargs_105697 = {}
        # Getting the type of 'assert_' (line 40)
        assert__105684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 40)
        assert__call_result_105698 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assert__105684, *[result_eq_105696], **kwargs_105697)
        
        
        # ################# End of 'test_sketch_dimensions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sketch_dimensions' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_105699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_105699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sketch_dimensions'
        return stypy_return_type_105699


    @norecursion
    def test_sketch_rows_norm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sketch_rows_norm'
        module_type_store = module_type_store.open_function_context('test_sketch_rows_norm', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_localization', localization)
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_function_name', 'TestClarksonWoodruffTransform.test_sketch_rows_norm')
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_param_names_list', [])
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestClarksonWoodruffTransform.test_sketch_rows_norm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestClarksonWoodruffTransform.test_sketch_rows_norm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sketch_rows_norm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sketch_rows_norm(...)' code ##################

        
        # Assigning a Num to a Name (line 47):
        int_105700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'int')
        # Assigning a type to the variable 'n_errors' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'n_errors', int_105700)
        
        # Assigning a List to a Name (line 49):
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_105701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        int_105702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105702)
        # Adding element type (line 49)
        int_105703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105703)
        # Adding element type (line 49)
        int_105704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105704)
        # Adding element type (line 49)
        int_105705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105705)
        # Adding element type (line 49)
        int_105706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105706)
        # Adding element type (line 49)
        int_105707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105707)
        # Adding element type (line 49)
        int_105708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105708)
        # Adding element type (line 49)
        int_105709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105709)
        # Adding element type (line 49)
        int_105710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105710)
        # Adding element type (line 49)
        int_105711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), list_105701, int_105711)
        
        # Assigning a type to the variable 'seeds' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'seeds', list_105701)
        
        # Getting the type of 'seeds' (line 52)
        seeds_105712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'seeds')
        # Testing the type of a for loop iterable (line 52)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 8), seeds_105712)
        # Getting the type of the for loop variable (line 52)
        for_loop_var_105713 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 8), seeds_105712)
        # Assigning a type to the variable 'seed_' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'seed_', for_loop_var_105713)
        # SSA begins for a for statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 53):
        
        # Call to clarkson_woodruff_transform(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'self' (line 53)
        self_105715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 49), 'self', False)
        # Obtaining the member 'dense_big_matrix' of a type (line 53)
        dense_big_matrix_105716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 49), self_105715, 'dense_big_matrix')
        # Getting the type of 'self' (line 54)
        self_105717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 49), 'self', False)
        # Obtaining the member 'n_sketch_rows' of a type (line 54)
        n_sketch_rows_105718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 49), self_105717, 'n_sketch_rows')
        # Getting the type of 'seed_' (line 54)
        seed__105719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 69), 'seed_', False)
        # Processing the call keyword arguments (line 53)
        kwargs_105720 = {}
        # Getting the type of 'clarkson_woodruff_transform' (line 53)
        clarkson_woodruff_transform_105714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'clarkson_woodruff_transform', False)
        # Calling clarkson_woodruff_transform(args, kwargs) (line 53)
        clarkson_woodruff_transform_call_result_105721 = invoke(stypy.reporting.localization.Localization(__file__, 53, 21), clarkson_woodruff_transform_105714, *[dense_big_matrix_105716, n_sketch_rows_105718, seed__105719], **kwargs_105720)
        
        # Assigning a type to the variable 'sketch' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'sketch', clarkson_woodruff_transform_call_result_105721)
        
        # Assigning a BinOp to a Name (line 57):
        
        # Call to norm(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'self' (line 57)
        self_105725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 33), 'self', False)
        # Obtaining the member 'dense_big_matrix' of a type (line 57)
        dense_big_matrix_105726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 33), self_105725, 'dense_big_matrix')
        # Processing the call keyword arguments (line 57)
        kwargs_105727 = {}
        # Getting the type of 'np' (line 57)
        np_105722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'np', False)
        # Obtaining the member 'linalg' of a type (line 57)
        linalg_105723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), np_105722, 'linalg')
        # Obtaining the member 'norm' of a type (line 57)
        norm_105724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), linalg_105723, 'norm')
        # Calling norm(args, kwargs) (line 57)
        norm_call_result_105728 = invoke(stypy.reporting.localization.Localization(__file__, 57, 18), norm_105724, *[dense_big_matrix_105726], **kwargs_105727)
        
        
        # Call to norm(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'sketch' (line 57)
        sketch_105732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 73), 'sketch', False)
        # Processing the call keyword arguments (line 57)
        kwargs_105733 = {}
        # Getting the type of 'np' (line 57)
        np_105729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 58), 'np', False)
        # Obtaining the member 'linalg' of a type (line 57)
        linalg_105730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 58), np_105729, 'linalg')
        # Obtaining the member 'norm' of a type (line 57)
        norm_105731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 58), linalg_105730, 'norm')
        # Calling norm(args, kwargs) (line 57)
        norm_call_result_105734 = invoke(stypy.reporting.localization.Localization(__file__, 57, 58), norm_105731, *[sketch_105732], **kwargs_105733)
        
        # Applying the binary operator '-' (line 57)
        result_sub_105735 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 18), '-', norm_call_result_105728, norm_call_result_105734)
        
        # Assigning a type to the variable 'err' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'err', result_sub_105735)
        
        
        # Getting the type of 'err' (line 58)
        err_105736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'err')
        # Getting the type of 'self' (line 58)
        self_105737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'self')
        # Obtaining the member 'threshold' of a type (line 58)
        threshold_105738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 21), self_105737, 'threshold')
        # Applying the binary operator '>' (line 58)
        result_gt_105739 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 15), '>', err_105736, threshold_105738)
        
        # Testing the type of an if condition (line 58)
        if_condition_105740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 12), result_gt_105739)
        # Assigning a type to the variable 'if_condition_105740' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'if_condition_105740', if_condition_105740)
        # SSA begins for if statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'n_errors' (line 59)
        n_errors_105741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'n_errors')
        int_105742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'int')
        # Applying the binary operator '+=' (line 59)
        result_iadd_105743 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 16), '+=', n_errors_105741, int_105742)
        # Assigning a type to the variable 'n_errors' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'n_errors', result_iadd_105743)
        
        # SSA join for if statement (line 58)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Getting the type of 'n_errors' (line 61)
        n_errors_105745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'n_errors', False)
        int_105746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'int')
        # Applying the binary operator '==' (line 61)
        result_eq_105747 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 16), '==', n_errors_105745, int_105746)
        
        # Processing the call keyword arguments (line 61)
        kwargs_105748 = {}
        # Getting the type of 'assert_' (line 61)
        assert__105744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 61)
        assert__call_result_105749 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert__105744, *[result_eq_105747], **kwargs_105748)
        
        
        # ################# End of 'test_sketch_rows_norm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sketch_rows_norm' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_105750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_105750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sketch_rows_norm'
        return stypy_return_type_105750


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestClarksonWoodruffTransform.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestClarksonWoodruffTransform' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'TestClarksonWoodruffTransform', TestClarksonWoodruffTransform)

# Assigning a Num to a Name (line 24):
int_105751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
# Getting the type of 'TestClarksonWoodruffTransform'
TestClarksonWoodruffTransform_105752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestClarksonWoodruffTransform')
# Setting the type of the member 'n_matrix_rows' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestClarksonWoodruffTransform_105752, 'n_matrix_rows', int_105751)

# Assigning a Num to a Name (line 25):
int_105753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'int')
# Getting the type of 'TestClarksonWoodruffTransform'
TestClarksonWoodruffTransform_105754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestClarksonWoodruffTransform')
# Setting the type of the member 'n_matrix_columns' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestClarksonWoodruffTransform_105754, 'n_matrix_columns', int_105753)

# Assigning a Num to a Name (line 28):
int_105755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'int')
# Getting the type of 'TestClarksonWoodruffTransform'
TestClarksonWoodruffTransform_105756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestClarksonWoodruffTransform')
# Setting the type of the member 'n_sketch_rows' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestClarksonWoodruffTransform_105756, 'n_sketch_rows', int_105755)

# Assigning a Num to a Name (line 31):
float_105757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'float')
# Getting the type of 'TestClarksonWoodruffTransform'
TestClarksonWoodruffTransform_105758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestClarksonWoodruffTransform')
# Setting the type of the member 'threshold' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestClarksonWoodruffTransform_105758, 'threshold', float_105757)

# Assigning a Call to a Name (line 33):

# Call to make_random_dense_gaussian_matrix(...): (line 33)
# Processing the call arguments (line 33)
# Getting the type of 'TestClarksonWoodruffTransform'
TestClarksonWoodruffTransform_105760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestClarksonWoodruffTransform', False)
# Obtaining the member 'n_matrix_rows' of a type
n_matrix_rows_105761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestClarksonWoodruffTransform_105760, 'n_matrix_rows')
# Getting the type of 'TestClarksonWoodruffTransform'
TestClarksonWoodruffTransform_105762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestClarksonWoodruffTransform', False)
# Obtaining the member 'n_matrix_columns' of a type
n_matrix_columns_105763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestClarksonWoodruffTransform_105762, 'n_matrix_columns')
# Processing the call keyword arguments (line 33)
kwargs_105764 = {}
# Getting the type of 'make_random_dense_gaussian_matrix' (line 33)
make_random_dense_gaussian_matrix_105759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'make_random_dense_gaussian_matrix', False)
# Calling make_random_dense_gaussian_matrix(args, kwargs) (line 33)
make_random_dense_gaussian_matrix_call_result_105765 = invoke(stypy.reporting.localization.Localization(__file__, 33, 23), make_random_dense_gaussian_matrix_105759, *[n_matrix_rows_105761, n_matrix_columns_105763], **kwargs_105764)

# Getting the type of 'TestClarksonWoodruffTransform'
TestClarksonWoodruffTransform_105766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestClarksonWoodruffTransform')
# Setting the type of the member 'dense_big_matrix' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestClarksonWoodruffTransform_105766, 'dense_big_matrix', make_random_dense_gaussian_matrix_call_result_105765)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
