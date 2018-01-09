
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import numpy as np
2: from numpy.testing import (assert_allclose,
3:                            assert_equal)
4: import scipy.linalg.cython_blas as blas
5: 
6: class TestDGEMM(object):
7:     
8:     def test_transposes(self):
9: 
10:         a = np.arange(12, dtype='d').reshape((3, 4))[:2,:2]
11:         b = np.arange(1, 13, dtype='d').reshape((4, 3))[:2,:2]
12:         c = np.empty((2, 4))[:2,:2]
13: 
14:         blas._test_dgemm(1., a, b, 0., c)
15:         assert_allclose(c, a.dot(b))
16: 
17:         blas._test_dgemm(1., a.T, b, 0., c)
18:         assert_allclose(c, a.T.dot(b))
19: 
20:         blas._test_dgemm(1., a, b.T, 0., c)
21:         assert_allclose(c, a.dot(b.T))
22: 
23:         blas._test_dgemm(1., a.T, b.T, 0., c)
24:         assert_allclose(c, a.T.dot(b.T))
25: 
26:         blas._test_dgemm(1., a, b, 0., c.T)
27:         assert_allclose(c, a.dot(b).T)
28: 
29:         blas._test_dgemm(1., a.T, b, 0., c.T)
30:         assert_allclose(c, a.T.dot(b).T)
31: 
32:         blas._test_dgemm(1., a, b.T, 0., c.T)
33:         assert_allclose(c, a.dot(b.T).T)
34: 
35:         blas._test_dgemm(1., a.T, b.T, 0., c.T)
36:         assert_allclose(c, a.T.dot(b.T).T)
37:     
38:     def test_shapes(self):
39:         a = np.arange(6, dtype='d').reshape((3, 2))
40:         b = np.arange(-6, 2, dtype='d').reshape((2, 4))
41:         c = np.empty((3, 4))
42: 
43:         blas._test_dgemm(1., a, b, 0., c)
44:         assert_allclose(c, a.dot(b))
45: 
46:         blas._test_dgemm(1., b.T, a.T, 0., c.T)
47:         assert_allclose(c, b.T.dot(a.T).T)
48:         
49: class TestWfuncPointers(object):
50:     ''' Test the function pointers that are expected to fail on
51:     Mac OS X without the additional entry statement in their definitions
52:     in fblas_l1.pyf.src. '''
53: 
54:     def test_complex_args(self):
55: 
56:         cx = np.array([.5 + 1.j, .25 - .375j, 12.5 - 4.j], np.complex64)
57:         cy = np.array([.8 + 2.j, .875 - .625j, -1. + 2.j], np.complex64)
58: 
59:         assert_allclose(blas._test_cdotc(cx, cy),
60:                         -17.6468753815+21.3718757629j, 5)
61:         assert_allclose(blas._test_cdotu(cx, cy),
62:                         -6.11562538147+30.3156242371j, 5)
63: 
64:         assert_equal(blas._test_icamax(cx), 3)
65: 
66:         assert_allclose(blas._test_scasum(cx), 18.625, 5)
67:         assert_allclose(blas._test_scnrm2(cx), 13.1796483994, 5)
68: 
69:         assert_allclose(blas._test_cdotc(cx[::2], cy[::2]),
70:                         -18.1000003815+21.2000007629j, 5)
71:         assert_allclose(blas._test_cdotu(cx[::2], cy[::2]),
72:                         -6.10000038147+30.7999992371j, 5)
73:         assert_allclose(blas._test_scasum(cx[::2]), 18., 5)
74:         assert_allclose(blas._test_scnrm2(cx[::2]), 13.1719398499, 5)
75:     
76:     def test_double_args(self):
77: 
78:         x = np.array([5., -3, -.5], np.float64)
79:         y = np.array([2, 1, .5], np.float64)
80: 
81:         assert_allclose(blas._test_dasum(x), 8.5, 10)
82:         assert_allclose(blas._test_ddot(x, y), 6.75, 10)
83:         assert_allclose(blas._test_dnrm2(x), 5.85234975815, 10)
84: 
85:         assert_allclose(blas._test_dasum(x[::2]), 5.5, 10)
86:         assert_allclose(blas._test_ddot(x[::2], y[::2]), 9.75, 10)
87:         assert_allclose(blas._test_dnrm2(x[::2]), 5.0249376297, 10)
88: 
89:         assert_equal(blas._test_idamax(x), 1)
90: 
91:     def test_float_args(self):
92: 
93:         x = np.array([5., -3, -.5], np.float32)
94:         y = np.array([2, 1, .5], np.float32)
95: 
96:         assert_equal(blas._test_isamax(x), 1)
97: 
98:         assert_allclose(blas._test_sasum(x), 8.5, 5)
99:         assert_allclose(blas._test_sdot(x, y), 6.75, 5)
100:         assert_allclose(blas._test_snrm2(x), 5.85234975815, 5)
101: 
102:         assert_allclose(blas._test_sasum(x[::2]), 5.5, 5)
103:         assert_allclose(blas._test_sdot(x[::2], y[::2]), 9.75, 5)
104:         assert_allclose(blas._test_snrm2(x[::2]), 5.0249376297, 5)
105: 
106:     def test_double_complex_args(self):
107: 
108:         cx = np.array([.5 + 1.j, .25 - .375j, 13. - 4.j], np.complex128)
109:         cy = np.array([.875 + 2.j, .875 - .625j, -1. + 2.j], np.complex128)
110: 
111:         assert_equal(blas._test_izamax(cx), 3)
112: 
113:         assert_allclose(blas._test_zdotc(cx, cy), -18.109375+22.296875j, 10)
114:         assert_allclose(blas._test_zdotu(cx, cy), -6.578125+31.390625j, 10)
115: 
116:         assert_allclose(blas._test_zdotc(cx[::2], cy[::2]),
117:                         -18.5625+22.125j, 10)
118:         assert_allclose(blas._test_zdotu(cx[::2], cy[::2]),
119:                         -6.5625+31.875j, 10)
120: 
121: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import numpy' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_53419 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy')

if (type(import_53419) is not StypyTypeError):

    if (import_53419 != 'pyd_module'):
        __import__(import_53419)
        sys_modules_53420 = sys.modules[import_53419]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'np', sys_modules_53420.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', import_53419)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from numpy.testing import assert_allclose, assert_equal' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_53421 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.testing')

if (type(import_53421) is not StypyTypeError):

    if (import_53421 != 'pyd_module'):
        __import__(import_53421)
        sys_modules_53422 = sys.modules[import_53421]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.testing', sys_modules_53422.module_type_store, module_type_store, ['assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_53422, sys_modules_53422.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal'], [assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.testing', import_53421)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import scipy.linalg.cython_blas' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_53423 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg.cython_blas')

if (type(import_53423) is not StypyTypeError):

    if (import_53423 != 'pyd_module'):
        __import__(import_53423)
        sys_modules_53424 = sys.modules[import_53423]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'blas', sys_modules_53424.module_type_store, module_type_store)
    else:
        import scipy.linalg.cython_blas as blas

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'blas', scipy.linalg.cython_blas, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg.cython_blas' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg.cython_blas', import_53423)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

# Declaration of the 'TestDGEMM' class

class TestDGEMM(object, ):

    @norecursion
    def test_transposes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_transposes'
        module_type_store = module_type_store.open_function_context('test_transposes', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_localization', localization)
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_function_name', 'TestDGEMM.test_transposes')
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_param_names_list', [])
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDGEMM.test_transposes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDGEMM.test_transposes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_transposes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_transposes(...)' code ##################

        
        # Assigning a Subscript to a Name (line 10):
        
        # Obtaining the type of the subscript
        int_53425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 54), 'int')
        slice_53426 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 10, 12), None, int_53425, None)
        int_53427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 57), 'int')
        slice_53428 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 10, 12), None, int_53427, None)
        
        # Call to reshape(...): (line 10)
        # Processing the call arguments (line 10)
        
        # Obtaining an instance of the builtin type 'tuple' (line 10)
        tuple_53437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 10)
        # Adding element type (line 10)
        int_53438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 46), tuple_53437, int_53438)
        # Adding element type (line 10)
        int_53439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 46), tuple_53437, int_53439)
        
        # Processing the call keyword arguments (line 10)
        kwargs_53440 = {}
        
        # Call to arange(...): (line 10)
        # Processing the call arguments (line 10)
        int_53431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 22), 'int')
        # Processing the call keyword arguments (line 10)
        str_53432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 32), 'str', 'd')
        keyword_53433 = str_53432
        kwargs_53434 = {'dtype': keyword_53433}
        # Getting the type of 'np' (line 10)
        np_53429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 10)
        arange_53430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), np_53429, 'arange')
        # Calling arange(args, kwargs) (line 10)
        arange_call_result_53435 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), arange_53430, *[int_53431], **kwargs_53434)
        
        # Obtaining the member 'reshape' of a type (line 10)
        reshape_53436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), arange_call_result_53435, 'reshape')
        # Calling reshape(args, kwargs) (line 10)
        reshape_call_result_53441 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), reshape_53436, *[tuple_53437], **kwargs_53440)
        
        # Obtaining the member '__getitem__' of a type (line 10)
        getitem___53442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), reshape_call_result_53441, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 10)
        subscript_call_result_53443 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), getitem___53442, (slice_53426, slice_53428))
        
        # Assigning a type to the variable 'a' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'a', subscript_call_result_53443)
        
        # Assigning a Subscript to a Name (line 11):
        
        # Obtaining the type of the subscript
        int_53444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 57), 'int')
        slice_53445 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 11, 12), None, int_53444, None)
        int_53446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 60), 'int')
        slice_53447 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 11, 12), None, int_53446, None)
        
        # Call to reshape(...): (line 11)
        # Processing the call arguments (line 11)
        
        # Obtaining an instance of the builtin type 'tuple' (line 11)
        tuple_53457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 11)
        # Adding element type (line 11)
        int_53458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 49), tuple_53457, int_53458)
        # Adding element type (line 11)
        int_53459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 49), tuple_53457, int_53459)
        
        # Processing the call keyword arguments (line 11)
        kwargs_53460 = {}
        
        # Call to arange(...): (line 11)
        # Processing the call arguments (line 11)
        int_53450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
        int_53451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 25), 'int')
        # Processing the call keyword arguments (line 11)
        str_53452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 35), 'str', 'd')
        keyword_53453 = str_53452
        kwargs_53454 = {'dtype': keyword_53453}
        # Getting the type of 'np' (line 11)
        np_53448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 11)
        arange_53449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), np_53448, 'arange')
        # Calling arange(args, kwargs) (line 11)
        arange_call_result_53455 = invoke(stypy.reporting.localization.Localization(__file__, 11, 12), arange_53449, *[int_53450, int_53451], **kwargs_53454)
        
        # Obtaining the member 'reshape' of a type (line 11)
        reshape_53456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), arange_call_result_53455, 'reshape')
        # Calling reshape(args, kwargs) (line 11)
        reshape_call_result_53461 = invoke(stypy.reporting.localization.Localization(__file__, 11, 12), reshape_53456, *[tuple_53457], **kwargs_53460)
        
        # Obtaining the member '__getitem__' of a type (line 11)
        getitem___53462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), reshape_call_result_53461, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 11)
        subscript_call_result_53463 = invoke(stypy.reporting.localization.Localization(__file__, 11, 12), getitem___53462, (slice_53445, slice_53447))
        
        # Assigning a type to the variable 'b' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'b', subscript_call_result_53463)
        
        # Assigning a Subscript to a Name (line 12):
        
        # Obtaining the type of the subscript
        int_53464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 30), 'int')
        slice_53465 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 12), None, int_53464, None)
        int_53466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 33), 'int')
        slice_53467 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 12), None, int_53466, None)
        
        # Call to empty(...): (line 12)
        # Processing the call arguments (line 12)
        
        # Obtaining an instance of the builtin type 'tuple' (line 12)
        tuple_53470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 12)
        # Adding element type (line 12)
        int_53471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 22), tuple_53470, int_53471)
        # Adding element type (line 12)
        int_53472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 22), tuple_53470, int_53472)
        
        # Processing the call keyword arguments (line 12)
        kwargs_53473 = {}
        # Getting the type of 'np' (line 12)
        np_53468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'np', False)
        # Obtaining the member 'empty' of a type (line 12)
        empty_53469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), np_53468, 'empty')
        # Calling empty(args, kwargs) (line 12)
        empty_call_result_53474 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), empty_53469, *[tuple_53470], **kwargs_53473)
        
        # Obtaining the member '__getitem__' of a type (line 12)
        getitem___53475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), empty_call_result_53474, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 12)
        subscript_call_result_53476 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), getitem___53475, (slice_53465, slice_53467))
        
        # Assigning a type to the variable 'c' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'c', subscript_call_result_53476)
        
        # Call to _test_dgemm(...): (line 14)
        # Processing the call arguments (line 14)
        float_53479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'float')
        # Getting the type of 'a' (line 14)
        a_53480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 29), 'a', False)
        # Getting the type of 'b' (line 14)
        b_53481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 32), 'b', False)
        float_53482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'float')
        # Getting the type of 'c' (line 14)
        c_53483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 39), 'c', False)
        # Processing the call keyword arguments (line 14)
        kwargs_53484 = {}
        # Getting the type of 'blas' (line 14)
        blas_53477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 14)
        _test_dgemm_53478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), blas_53477, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 14)
        _test_dgemm_call_result_53485 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), _test_dgemm_53478, *[float_53479, a_53480, b_53481, float_53482, c_53483], **kwargs_53484)
        
        
        # Call to assert_allclose(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'c' (line 15)
        c_53487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'c', False)
        
        # Call to dot(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'b' (line 15)
        b_53490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 33), 'b', False)
        # Processing the call keyword arguments (line 15)
        kwargs_53491 = {}
        # Getting the type of 'a' (line 15)
        a_53488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'a', False)
        # Obtaining the member 'dot' of a type (line 15)
        dot_53489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 27), a_53488, 'dot')
        # Calling dot(args, kwargs) (line 15)
        dot_call_result_53492 = invoke(stypy.reporting.localization.Localization(__file__, 15, 27), dot_53489, *[b_53490], **kwargs_53491)
        
        # Processing the call keyword arguments (line 15)
        kwargs_53493 = {}
        # Getting the type of 'assert_allclose' (line 15)
        assert_allclose_53486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 15)
        assert_allclose_call_result_53494 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), assert_allclose_53486, *[c_53487, dot_call_result_53492], **kwargs_53493)
        
        
        # Call to _test_dgemm(...): (line 17)
        # Processing the call arguments (line 17)
        float_53497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'float')
        # Getting the type of 'a' (line 17)
        a_53498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'a', False)
        # Obtaining the member 'T' of a type (line 17)
        T_53499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 29), a_53498, 'T')
        # Getting the type of 'b' (line 17)
        b_53500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 34), 'b', False)
        float_53501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 37), 'float')
        # Getting the type of 'c' (line 17)
        c_53502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 41), 'c', False)
        # Processing the call keyword arguments (line 17)
        kwargs_53503 = {}
        # Getting the type of 'blas' (line 17)
        blas_53495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 17)
        _test_dgemm_53496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), blas_53495, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 17)
        _test_dgemm_call_result_53504 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), _test_dgemm_53496, *[float_53497, T_53499, b_53500, float_53501, c_53502], **kwargs_53503)
        
        
        # Call to assert_allclose(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'c' (line 18)
        c_53506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'c', False)
        
        # Call to dot(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'b' (line 18)
        b_53510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 35), 'b', False)
        # Processing the call keyword arguments (line 18)
        kwargs_53511 = {}
        # Getting the type of 'a' (line 18)
        a_53507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 'a', False)
        # Obtaining the member 'T' of a type (line 18)
        T_53508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 27), a_53507, 'T')
        # Obtaining the member 'dot' of a type (line 18)
        dot_53509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 27), T_53508, 'dot')
        # Calling dot(args, kwargs) (line 18)
        dot_call_result_53512 = invoke(stypy.reporting.localization.Localization(__file__, 18, 27), dot_53509, *[b_53510], **kwargs_53511)
        
        # Processing the call keyword arguments (line 18)
        kwargs_53513 = {}
        # Getting the type of 'assert_allclose' (line 18)
        assert_allclose_53505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 18)
        assert_allclose_call_result_53514 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), assert_allclose_53505, *[c_53506, dot_call_result_53512], **kwargs_53513)
        
        
        # Call to _test_dgemm(...): (line 20)
        # Processing the call arguments (line 20)
        float_53517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'float')
        # Getting the type of 'a' (line 20)
        a_53518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 'a', False)
        # Getting the type of 'b' (line 20)
        b_53519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 32), 'b', False)
        # Obtaining the member 'T' of a type (line 20)
        T_53520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 32), b_53519, 'T')
        float_53521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 37), 'float')
        # Getting the type of 'c' (line 20)
        c_53522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 41), 'c', False)
        # Processing the call keyword arguments (line 20)
        kwargs_53523 = {}
        # Getting the type of 'blas' (line 20)
        blas_53515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 20)
        _test_dgemm_53516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), blas_53515, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 20)
        _test_dgemm_call_result_53524 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), _test_dgemm_53516, *[float_53517, a_53518, T_53520, float_53521, c_53522], **kwargs_53523)
        
        
        # Call to assert_allclose(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'c' (line 21)
        c_53526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'c', False)
        
        # Call to dot(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'b' (line 21)
        b_53529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 33), 'b', False)
        # Obtaining the member 'T' of a type (line 21)
        T_53530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 33), b_53529, 'T')
        # Processing the call keyword arguments (line 21)
        kwargs_53531 = {}
        # Getting the type of 'a' (line 21)
        a_53527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'a', False)
        # Obtaining the member 'dot' of a type (line 21)
        dot_53528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 27), a_53527, 'dot')
        # Calling dot(args, kwargs) (line 21)
        dot_call_result_53532 = invoke(stypy.reporting.localization.Localization(__file__, 21, 27), dot_53528, *[T_53530], **kwargs_53531)
        
        # Processing the call keyword arguments (line 21)
        kwargs_53533 = {}
        # Getting the type of 'assert_allclose' (line 21)
        assert_allclose_53525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 21)
        assert_allclose_call_result_53534 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert_allclose_53525, *[c_53526, dot_call_result_53532], **kwargs_53533)
        
        
        # Call to _test_dgemm(...): (line 23)
        # Processing the call arguments (line 23)
        float_53537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'float')
        # Getting the type of 'a' (line 23)
        a_53538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'a', False)
        # Obtaining the member 'T' of a type (line 23)
        T_53539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 29), a_53538, 'T')
        # Getting the type of 'b' (line 23)
        b_53540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'b', False)
        # Obtaining the member 'T' of a type (line 23)
        T_53541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 34), b_53540, 'T')
        float_53542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 39), 'float')
        # Getting the type of 'c' (line 23)
        c_53543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 43), 'c', False)
        # Processing the call keyword arguments (line 23)
        kwargs_53544 = {}
        # Getting the type of 'blas' (line 23)
        blas_53535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 23)
        _test_dgemm_53536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), blas_53535, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 23)
        _test_dgemm_call_result_53545 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), _test_dgemm_53536, *[float_53537, T_53539, T_53541, float_53542, c_53543], **kwargs_53544)
        
        
        # Call to assert_allclose(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'c' (line 24)
        c_53547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'c', False)
        
        # Call to dot(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'b' (line 24)
        b_53551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 35), 'b', False)
        # Obtaining the member 'T' of a type (line 24)
        T_53552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 35), b_53551, 'T')
        # Processing the call keyword arguments (line 24)
        kwargs_53553 = {}
        # Getting the type of 'a' (line 24)
        a_53548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'a', False)
        # Obtaining the member 'T' of a type (line 24)
        T_53549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 27), a_53548, 'T')
        # Obtaining the member 'dot' of a type (line 24)
        dot_53550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 27), T_53549, 'dot')
        # Calling dot(args, kwargs) (line 24)
        dot_call_result_53554 = invoke(stypy.reporting.localization.Localization(__file__, 24, 27), dot_53550, *[T_53552], **kwargs_53553)
        
        # Processing the call keyword arguments (line 24)
        kwargs_53555 = {}
        # Getting the type of 'assert_allclose' (line 24)
        assert_allclose_53546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 24)
        assert_allclose_call_result_53556 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), assert_allclose_53546, *[c_53547, dot_call_result_53554], **kwargs_53555)
        
        
        # Call to _test_dgemm(...): (line 26)
        # Processing the call arguments (line 26)
        float_53559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'float')
        # Getting the type of 'a' (line 26)
        a_53560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'a', False)
        # Getting the type of 'b' (line 26)
        b_53561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 32), 'b', False)
        float_53562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'float')
        # Getting the type of 'c' (line 26)
        c_53563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 39), 'c', False)
        # Obtaining the member 'T' of a type (line 26)
        T_53564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 39), c_53563, 'T')
        # Processing the call keyword arguments (line 26)
        kwargs_53565 = {}
        # Getting the type of 'blas' (line 26)
        blas_53557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 26)
        _test_dgemm_53558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), blas_53557, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 26)
        _test_dgemm_call_result_53566 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), _test_dgemm_53558, *[float_53559, a_53560, b_53561, float_53562, T_53564], **kwargs_53565)
        
        
        # Call to assert_allclose(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'c' (line 27)
        c_53568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'c', False)
        
        # Call to dot(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'b' (line 27)
        b_53571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'b', False)
        # Processing the call keyword arguments (line 27)
        kwargs_53572 = {}
        # Getting the type of 'a' (line 27)
        a_53569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'a', False)
        # Obtaining the member 'dot' of a type (line 27)
        dot_53570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 27), a_53569, 'dot')
        # Calling dot(args, kwargs) (line 27)
        dot_call_result_53573 = invoke(stypy.reporting.localization.Localization(__file__, 27, 27), dot_53570, *[b_53571], **kwargs_53572)
        
        # Obtaining the member 'T' of a type (line 27)
        T_53574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 27), dot_call_result_53573, 'T')
        # Processing the call keyword arguments (line 27)
        kwargs_53575 = {}
        # Getting the type of 'assert_allclose' (line 27)
        assert_allclose_53567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 27)
        assert_allclose_call_result_53576 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_allclose_53567, *[c_53568, T_53574], **kwargs_53575)
        
        
        # Call to _test_dgemm(...): (line 29)
        # Processing the call arguments (line 29)
        float_53579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'float')
        # Getting the type of 'a' (line 29)
        a_53580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'a', False)
        # Obtaining the member 'T' of a type (line 29)
        T_53581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 29), a_53580, 'T')
        # Getting the type of 'b' (line 29)
        b_53582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 34), 'b', False)
        float_53583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 37), 'float')
        # Getting the type of 'c' (line 29)
        c_53584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 41), 'c', False)
        # Obtaining the member 'T' of a type (line 29)
        T_53585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 41), c_53584, 'T')
        # Processing the call keyword arguments (line 29)
        kwargs_53586 = {}
        # Getting the type of 'blas' (line 29)
        blas_53577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 29)
        _test_dgemm_53578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), blas_53577, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 29)
        _test_dgemm_call_result_53587 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), _test_dgemm_53578, *[float_53579, T_53581, b_53582, float_53583, T_53585], **kwargs_53586)
        
        
        # Call to assert_allclose(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'c' (line 30)
        c_53589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'c', False)
        
        # Call to dot(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'b' (line 30)
        b_53593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'b', False)
        # Processing the call keyword arguments (line 30)
        kwargs_53594 = {}
        # Getting the type of 'a' (line 30)
        a_53590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'a', False)
        # Obtaining the member 'T' of a type (line 30)
        T_53591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 27), a_53590, 'T')
        # Obtaining the member 'dot' of a type (line 30)
        dot_53592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 27), T_53591, 'dot')
        # Calling dot(args, kwargs) (line 30)
        dot_call_result_53595 = invoke(stypy.reporting.localization.Localization(__file__, 30, 27), dot_53592, *[b_53593], **kwargs_53594)
        
        # Obtaining the member 'T' of a type (line 30)
        T_53596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 27), dot_call_result_53595, 'T')
        # Processing the call keyword arguments (line 30)
        kwargs_53597 = {}
        # Getting the type of 'assert_allclose' (line 30)
        assert_allclose_53588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 30)
        assert_allclose_call_result_53598 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_allclose_53588, *[c_53589, T_53596], **kwargs_53597)
        
        
        # Call to _test_dgemm(...): (line 32)
        # Processing the call arguments (line 32)
        float_53601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'float')
        # Getting the type of 'a' (line 32)
        a_53602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'a', False)
        # Getting the type of 'b' (line 32)
        b_53603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'b', False)
        # Obtaining the member 'T' of a type (line 32)
        T_53604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 32), b_53603, 'T')
        float_53605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'float')
        # Getting the type of 'c' (line 32)
        c_53606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 41), 'c', False)
        # Obtaining the member 'T' of a type (line 32)
        T_53607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 41), c_53606, 'T')
        # Processing the call keyword arguments (line 32)
        kwargs_53608 = {}
        # Getting the type of 'blas' (line 32)
        blas_53599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 32)
        _test_dgemm_53600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), blas_53599, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 32)
        _test_dgemm_call_result_53609 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), _test_dgemm_53600, *[float_53601, a_53602, T_53604, float_53605, T_53607], **kwargs_53608)
        
        
        # Call to assert_allclose(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'c' (line 33)
        c_53611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'c', False)
        
        # Call to dot(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'b' (line 33)
        b_53614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'b', False)
        # Obtaining the member 'T' of a type (line 33)
        T_53615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 33), b_53614, 'T')
        # Processing the call keyword arguments (line 33)
        kwargs_53616 = {}
        # Getting the type of 'a' (line 33)
        a_53612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'a', False)
        # Obtaining the member 'dot' of a type (line 33)
        dot_53613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 27), a_53612, 'dot')
        # Calling dot(args, kwargs) (line 33)
        dot_call_result_53617 = invoke(stypy.reporting.localization.Localization(__file__, 33, 27), dot_53613, *[T_53615], **kwargs_53616)
        
        # Obtaining the member 'T' of a type (line 33)
        T_53618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 27), dot_call_result_53617, 'T')
        # Processing the call keyword arguments (line 33)
        kwargs_53619 = {}
        # Getting the type of 'assert_allclose' (line 33)
        assert_allclose_53610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 33)
        assert_allclose_call_result_53620 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assert_allclose_53610, *[c_53611, T_53618], **kwargs_53619)
        
        
        # Call to _test_dgemm(...): (line 35)
        # Processing the call arguments (line 35)
        float_53623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'float')
        # Getting the type of 'a' (line 35)
        a_53624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'a', False)
        # Obtaining the member 'T' of a type (line 35)
        T_53625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 29), a_53624, 'T')
        # Getting the type of 'b' (line 35)
        b_53626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'b', False)
        # Obtaining the member 'T' of a type (line 35)
        T_53627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), b_53626, 'T')
        float_53628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 39), 'float')
        # Getting the type of 'c' (line 35)
        c_53629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 'c', False)
        # Obtaining the member 'T' of a type (line 35)
        T_53630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 43), c_53629, 'T')
        # Processing the call keyword arguments (line 35)
        kwargs_53631 = {}
        # Getting the type of 'blas' (line 35)
        blas_53621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 35)
        _test_dgemm_53622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), blas_53621, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 35)
        _test_dgemm_call_result_53632 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), _test_dgemm_53622, *[float_53623, T_53625, T_53627, float_53628, T_53630], **kwargs_53631)
        
        
        # Call to assert_allclose(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'c' (line 36)
        c_53634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'c', False)
        
        # Call to dot(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'b' (line 36)
        b_53638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 35), 'b', False)
        # Obtaining the member 'T' of a type (line 36)
        T_53639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), b_53638, 'T')
        # Processing the call keyword arguments (line 36)
        kwargs_53640 = {}
        # Getting the type of 'a' (line 36)
        a_53635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'a', False)
        # Obtaining the member 'T' of a type (line 36)
        T_53636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), a_53635, 'T')
        # Obtaining the member 'dot' of a type (line 36)
        dot_53637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), T_53636, 'dot')
        # Calling dot(args, kwargs) (line 36)
        dot_call_result_53641 = invoke(stypy.reporting.localization.Localization(__file__, 36, 27), dot_53637, *[T_53639], **kwargs_53640)
        
        # Obtaining the member 'T' of a type (line 36)
        T_53642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), dot_call_result_53641, 'T')
        # Processing the call keyword arguments (line 36)
        kwargs_53643 = {}
        # Getting the type of 'assert_allclose' (line 36)
        assert_allclose_53633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 36)
        assert_allclose_call_result_53644 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert_allclose_53633, *[c_53634, T_53642], **kwargs_53643)
        
        
        # ################# End of 'test_transposes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_transposes' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_53645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53645)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_transposes'
        return stypy_return_type_53645


    @norecursion
    def test_shapes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_shapes'
        module_type_store = module_type_store.open_function_context('test_shapes', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_localization', localization)
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_function_name', 'TestDGEMM.test_shapes')
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_param_names_list', [])
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDGEMM.test_shapes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDGEMM.test_shapes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_shapes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_shapes(...)' code ##################

        
        # Assigning a Call to a Name (line 39):
        
        # Call to reshape(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_53654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_53655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 45), tuple_53654, int_53655)
        # Adding element type (line 39)
        int_53656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 45), tuple_53654, int_53656)
        
        # Processing the call keyword arguments (line 39)
        kwargs_53657 = {}
        
        # Call to arange(...): (line 39)
        # Processing the call arguments (line 39)
        int_53648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'int')
        # Processing the call keyword arguments (line 39)
        str_53649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 31), 'str', 'd')
        keyword_53650 = str_53649
        kwargs_53651 = {'dtype': keyword_53650}
        # Getting the type of 'np' (line 39)
        np_53646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 39)
        arange_53647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), np_53646, 'arange')
        # Calling arange(args, kwargs) (line 39)
        arange_call_result_53652 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), arange_53647, *[int_53648], **kwargs_53651)
        
        # Obtaining the member 'reshape' of a type (line 39)
        reshape_53653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), arange_call_result_53652, 'reshape')
        # Calling reshape(args, kwargs) (line 39)
        reshape_call_result_53658 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), reshape_53653, *[tuple_53654], **kwargs_53657)
        
        # Assigning a type to the variable 'a' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'a', reshape_call_result_53658)
        
        # Assigning a Call to a Name (line 40):
        
        # Call to reshape(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_53668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        int_53669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 49), tuple_53668, int_53669)
        # Adding element type (line 40)
        int_53670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 49), tuple_53668, int_53670)
        
        # Processing the call keyword arguments (line 40)
        kwargs_53671 = {}
        
        # Call to arange(...): (line 40)
        # Processing the call arguments (line 40)
        int_53661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'int')
        int_53662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
        # Processing the call keyword arguments (line 40)
        str_53663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'str', 'd')
        keyword_53664 = str_53663
        kwargs_53665 = {'dtype': keyword_53664}
        # Getting the type of 'np' (line 40)
        np_53659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 40)
        arange_53660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), np_53659, 'arange')
        # Calling arange(args, kwargs) (line 40)
        arange_call_result_53666 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), arange_53660, *[int_53661, int_53662], **kwargs_53665)
        
        # Obtaining the member 'reshape' of a type (line 40)
        reshape_53667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), arange_call_result_53666, 'reshape')
        # Calling reshape(args, kwargs) (line 40)
        reshape_call_result_53672 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), reshape_53667, *[tuple_53668], **kwargs_53671)
        
        # Assigning a type to the variable 'b' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'b', reshape_call_result_53672)
        
        # Assigning a Call to a Name (line 41):
        
        # Call to empty(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_53675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        int_53676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), tuple_53675, int_53676)
        # Adding element type (line 41)
        int_53677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), tuple_53675, int_53677)
        
        # Processing the call keyword arguments (line 41)
        kwargs_53678 = {}
        # Getting the type of 'np' (line 41)
        np_53673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'np', False)
        # Obtaining the member 'empty' of a type (line 41)
        empty_53674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), np_53673, 'empty')
        # Calling empty(args, kwargs) (line 41)
        empty_call_result_53679 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), empty_53674, *[tuple_53675], **kwargs_53678)
        
        # Assigning a type to the variable 'c' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'c', empty_call_result_53679)
        
        # Call to _test_dgemm(...): (line 43)
        # Processing the call arguments (line 43)
        float_53682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'float')
        # Getting the type of 'a' (line 43)
        a_53683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 29), 'a', False)
        # Getting the type of 'b' (line 43)
        b_53684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'b', False)
        float_53685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 35), 'float')
        # Getting the type of 'c' (line 43)
        c_53686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 39), 'c', False)
        # Processing the call keyword arguments (line 43)
        kwargs_53687 = {}
        # Getting the type of 'blas' (line 43)
        blas_53680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 43)
        _test_dgemm_53681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), blas_53680, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 43)
        _test_dgemm_call_result_53688 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), _test_dgemm_53681, *[float_53682, a_53683, b_53684, float_53685, c_53686], **kwargs_53687)
        
        
        # Call to assert_allclose(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'c' (line 44)
        c_53690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'c', False)
        
        # Call to dot(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'b' (line 44)
        b_53693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'b', False)
        # Processing the call keyword arguments (line 44)
        kwargs_53694 = {}
        # Getting the type of 'a' (line 44)
        a_53691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'a', False)
        # Obtaining the member 'dot' of a type (line 44)
        dot_53692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), a_53691, 'dot')
        # Calling dot(args, kwargs) (line 44)
        dot_call_result_53695 = invoke(stypy.reporting.localization.Localization(__file__, 44, 27), dot_53692, *[b_53693], **kwargs_53694)
        
        # Processing the call keyword arguments (line 44)
        kwargs_53696 = {}
        # Getting the type of 'assert_allclose' (line 44)
        assert_allclose_53689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 44)
        assert_allclose_call_result_53697 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), assert_allclose_53689, *[c_53690, dot_call_result_53695], **kwargs_53696)
        
        
        # Call to _test_dgemm(...): (line 46)
        # Processing the call arguments (line 46)
        float_53700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'float')
        # Getting the type of 'b' (line 46)
        b_53701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'b', False)
        # Obtaining the member 'T' of a type (line 46)
        T_53702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 29), b_53701, 'T')
        # Getting the type of 'a' (line 46)
        a_53703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'a', False)
        # Obtaining the member 'T' of a type (line 46)
        T_53704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 34), a_53703, 'T')
        float_53705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'float')
        # Getting the type of 'c' (line 46)
        c_53706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 43), 'c', False)
        # Obtaining the member 'T' of a type (line 46)
        T_53707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 43), c_53706, 'T')
        # Processing the call keyword arguments (line 46)
        kwargs_53708 = {}
        # Getting the type of 'blas' (line 46)
        blas_53698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'blas', False)
        # Obtaining the member '_test_dgemm' of a type (line 46)
        _test_dgemm_53699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), blas_53698, '_test_dgemm')
        # Calling _test_dgemm(args, kwargs) (line 46)
        _test_dgemm_call_result_53709 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), _test_dgemm_53699, *[float_53700, T_53702, T_53704, float_53705, T_53707], **kwargs_53708)
        
        
        # Call to assert_allclose(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'c' (line 47)
        c_53711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'c', False)
        
        # Call to dot(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'a' (line 47)
        a_53715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 35), 'a', False)
        # Obtaining the member 'T' of a type (line 47)
        T_53716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 35), a_53715, 'T')
        # Processing the call keyword arguments (line 47)
        kwargs_53717 = {}
        # Getting the type of 'b' (line 47)
        b_53712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'b', False)
        # Obtaining the member 'T' of a type (line 47)
        T_53713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 27), b_53712, 'T')
        # Obtaining the member 'dot' of a type (line 47)
        dot_53714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 27), T_53713, 'dot')
        # Calling dot(args, kwargs) (line 47)
        dot_call_result_53718 = invoke(stypy.reporting.localization.Localization(__file__, 47, 27), dot_53714, *[T_53716], **kwargs_53717)
        
        # Obtaining the member 'T' of a type (line 47)
        T_53719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 27), dot_call_result_53718, 'T')
        # Processing the call keyword arguments (line 47)
        kwargs_53720 = {}
        # Getting the type of 'assert_allclose' (line 47)
        assert_allclose_53710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 47)
        assert_allclose_call_result_53721 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_allclose_53710, *[c_53711, T_53719], **kwargs_53720)
        
        
        # ################# End of 'test_shapes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_shapes' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_53722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53722)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_shapes'
        return stypy_return_type_53722


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 0, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDGEMM.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDGEMM' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'TestDGEMM', TestDGEMM)
# Declaration of the 'TestWfuncPointers' class

class TestWfuncPointers(object, ):
    str_53723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', ' Test the function pointers that are expected to fail on\n    Mac OS X without the additional entry statement in their definitions\n    in fblas_l1.pyf.src. ')

    @norecursion
    def test_complex_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_complex_args'
        module_type_store = module_type_store.open_function_context('test_complex_args', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_localization', localization)
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_function_name', 'TestWfuncPointers.test_complex_args')
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWfuncPointers.test_complex_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWfuncPointers.test_complex_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_complex_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_complex_args(...)' code ##################

        
        # Assigning a Call to a Name (line 56):
        
        # Call to array(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_53726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        float_53727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'float')
        complex_53728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'complex')
        # Applying the binary operator '+' (line 56)
        result_add_53729 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 23), '+', float_53727, complex_53728)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 22), list_53726, result_add_53729)
        # Adding element type (line 56)
        float_53730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'float')
        complex_53731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'complex')
        # Applying the binary operator '-' (line 56)
        result_sub_53732 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 33), '-', float_53730, complex_53731)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 22), list_53726, result_sub_53732)
        # Adding element type (line 56)
        float_53733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 46), 'float')
        complex_53734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 53), 'complex')
        # Applying the binary operator '-' (line 56)
        result_sub_53735 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 46), '-', float_53733, complex_53734)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 22), list_53726, result_sub_53735)
        
        # Getting the type of 'np' (line 56)
        np_53736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 59), 'np', False)
        # Obtaining the member 'complex64' of a type (line 56)
        complex64_53737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 59), np_53736, 'complex64')
        # Processing the call keyword arguments (line 56)
        kwargs_53738 = {}
        # Getting the type of 'np' (line 56)
        np_53724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 56)
        array_53725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 13), np_53724, 'array')
        # Calling array(args, kwargs) (line 56)
        array_call_result_53739 = invoke(stypy.reporting.localization.Localization(__file__, 56, 13), array_53725, *[list_53726, complex64_53737], **kwargs_53738)
        
        # Assigning a type to the variable 'cx' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'cx', array_call_result_53739)
        
        # Assigning a Call to a Name (line 57):
        
        # Call to array(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_53742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        float_53743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'float')
        complex_53744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'complex')
        # Applying the binary operator '+' (line 57)
        result_add_53745 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 23), '+', float_53743, complex_53744)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 22), list_53742, result_add_53745)
        # Adding element type (line 57)
        float_53746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 33), 'float')
        complex_53747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 40), 'complex')
        # Applying the binary operator '-' (line 57)
        result_sub_53748 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 33), '-', float_53746, complex_53747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 22), list_53742, result_sub_53748)
        # Adding element type (line 57)
        float_53749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 47), 'float')
        complex_53750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 53), 'complex')
        # Applying the binary operator '+' (line 57)
        result_add_53751 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 47), '+', float_53749, complex_53750)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 22), list_53742, result_add_53751)
        
        # Getting the type of 'np' (line 57)
        np_53752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 59), 'np', False)
        # Obtaining the member 'complex64' of a type (line 57)
        complex64_53753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 59), np_53752, 'complex64')
        # Processing the call keyword arguments (line 57)
        kwargs_53754 = {}
        # Getting the type of 'np' (line 57)
        np_53740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 57)
        array_53741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), np_53740, 'array')
        # Calling array(args, kwargs) (line 57)
        array_call_result_53755 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), array_53741, *[list_53742, complex64_53753], **kwargs_53754)
        
        # Assigning a type to the variable 'cy' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'cy', array_call_result_53755)
        
        # Call to assert_allclose(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Call to _test_cdotc(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'cx' (line 59)
        cx_53759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 41), 'cx', False)
        # Getting the type of 'cy' (line 59)
        cy_53760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'cy', False)
        # Processing the call keyword arguments (line 59)
        kwargs_53761 = {}
        # Getting the type of 'blas' (line 59)
        blas_53757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'blas', False)
        # Obtaining the member '_test_cdotc' of a type (line 59)
        _test_cdotc_53758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), blas_53757, '_test_cdotc')
        # Calling _test_cdotc(args, kwargs) (line 59)
        _test_cdotc_call_result_53762 = invoke(stypy.reporting.localization.Localization(__file__, 59, 24), _test_cdotc_53758, *[cx_53759, cy_53760], **kwargs_53761)
        
        float_53763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'float')
        complex_53764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'complex')
        # Applying the binary operator '+' (line 60)
        result_add_53765 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 24), '+', float_53763, complex_53764)
        
        int_53766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 55), 'int')
        # Processing the call keyword arguments (line 59)
        kwargs_53767 = {}
        # Getting the type of 'assert_allclose' (line 59)
        assert_allclose_53756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 59)
        assert_allclose_call_result_53768 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert_allclose_53756, *[_test_cdotc_call_result_53762, result_add_53765, int_53766], **kwargs_53767)
        
        
        # Call to assert_allclose(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to _test_cdotu(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'cx' (line 61)
        cx_53772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'cx', False)
        # Getting the type of 'cy' (line 61)
        cy_53773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'cy', False)
        # Processing the call keyword arguments (line 61)
        kwargs_53774 = {}
        # Getting the type of 'blas' (line 61)
        blas_53770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'blas', False)
        # Obtaining the member '_test_cdotu' of a type (line 61)
        _test_cdotu_53771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), blas_53770, '_test_cdotu')
        # Calling _test_cdotu(args, kwargs) (line 61)
        _test_cdotu_call_result_53775 = invoke(stypy.reporting.localization.Localization(__file__, 61, 24), _test_cdotu_53771, *[cx_53772, cy_53773], **kwargs_53774)
        
        float_53776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'float')
        complex_53777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 39), 'complex')
        # Applying the binary operator '+' (line 62)
        result_add_53778 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 24), '+', float_53776, complex_53777)
        
        int_53779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 55), 'int')
        # Processing the call keyword arguments (line 61)
        kwargs_53780 = {}
        # Getting the type of 'assert_allclose' (line 61)
        assert_allclose_53769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 61)
        assert_allclose_call_result_53781 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_allclose_53769, *[_test_cdotu_call_result_53775, result_add_53778, int_53779], **kwargs_53780)
        
        
        # Call to assert_equal(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to _test_icamax(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'cx' (line 64)
        cx_53785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'cx', False)
        # Processing the call keyword arguments (line 64)
        kwargs_53786 = {}
        # Getting the type of 'blas' (line 64)
        blas_53783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'blas', False)
        # Obtaining the member '_test_icamax' of a type (line 64)
        _test_icamax_53784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), blas_53783, '_test_icamax')
        # Calling _test_icamax(args, kwargs) (line 64)
        _test_icamax_call_result_53787 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), _test_icamax_53784, *[cx_53785], **kwargs_53786)
        
        int_53788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 44), 'int')
        # Processing the call keyword arguments (line 64)
        kwargs_53789 = {}
        # Getting the type of 'assert_equal' (line 64)
        assert_equal_53782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 64)
        assert_equal_call_result_53790 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_equal_53782, *[_test_icamax_call_result_53787, int_53788], **kwargs_53789)
        
        
        # Call to assert_allclose(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to _test_scasum(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'cx' (line 66)
        cx_53794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'cx', False)
        # Processing the call keyword arguments (line 66)
        kwargs_53795 = {}
        # Getting the type of 'blas' (line 66)
        blas_53792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'blas', False)
        # Obtaining the member '_test_scasum' of a type (line 66)
        _test_scasum_53793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 24), blas_53792, '_test_scasum')
        # Calling _test_scasum(args, kwargs) (line 66)
        _test_scasum_call_result_53796 = invoke(stypy.reporting.localization.Localization(__file__, 66, 24), _test_scasum_53793, *[cx_53794], **kwargs_53795)
        
        float_53797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 47), 'float')
        int_53798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 55), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_53799 = {}
        # Getting the type of 'assert_allclose' (line 66)
        assert_allclose_53791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 66)
        assert_allclose_call_result_53800 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_allclose_53791, *[_test_scasum_call_result_53796, float_53797, int_53798], **kwargs_53799)
        
        
        # Call to assert_allclose(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to _test_scnrm2(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'cx' (line 67)
        cx_53804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 42), 'cx', False)
        # Processing the call keyword arguments (line 67)
        kwargs_53805 = {}
        # Getting the type of 'blas' (line 67)
        blas_53802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'blas', False)
        # Obtaining the member '_test_scnrm2' of a type (line 67)
        _test_scnrm2_53803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 24), blas_53802, '_test_scnrm2')
        # Calling _test_scnrm2(args, kwargs) (line 67)
        _test_scnrm2_call_result_53806 = invoke(stypy.reporting.localization.Localization(__file__, 67, 24), _test_scnrm2_53803, *[cx_53804], **kwargs_53805)
        
        float_53807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 47), 'float')
        int_53808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 62), 'int')
        # Processing the call keyword arguments (line 67)
        kwargs_53809 = {}
        # Getting the type of 'assert_allclose' (line 67)
        assert_allclose_53801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 67)
        assert_allclose_call_result_53810 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assert_allclose_53801, *[_test_scnrm2_call_result_53806, float_53807, int_53808], **kwargs_53809)
        
        
        # Call to assert_allclose(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to _test_cdotc(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining the type of the subscript
        int_53814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 46), 'int')
        slice_53815 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 41), None, None, int_53814)
        # Getting the type of 'cx' (line 69)
        cx_53816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 41), 'cx', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___53817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 41), cx_53816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_53818 = invoke(stypy.reporting.localization.Localization(__file__, 69, 41), getitem___53817, slice_53815)
        
        
        # Obtaining the type of the subscript
        int_53819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 55), 'int')
        slice_53820 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 50), None, None, int_53819)
        # Getting the type of 'cy' (line 69)
        cy_53821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 50), 'cy', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___53822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 50), cy_53821, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_53823 = invoke(stypy.reporting.localization.Localization(__file__, 69, 50), getitem___53822, slice_53820)
        
        # Processing the call keyword arguments (line 69)
        kwargs_53824 = {}
        # Getting the type of 'blas' (line 69)
        blas_53812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'blas', False)
        # Obtaining the member '_test_cdotc' of a type (line 69)
        _test_cdotc_53813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 24), blas_53812, '_test_cdotc')
        # Calling _test_cdotc(args, kwargs) (line 69)
        _test_cdotc_call_result_53825 = invoke(stypy.reporting.localization.Localization(__file__, 69, 24), _test_cdotc_53813, *[subscript_call_result_53818, subscript_call_result_53823], **kwargs_53824)
        
        float_53826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 24), 'float')
        complex_53827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 39), 'complex')
        # Applying the binary operator '+' (line 70)
        result_add_53828 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 24), '+', float_53826, complex_53827)
        
        int_53829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 55), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_53830 = {}
        # Getting the type of 'assert_allclose' (line 69)
        assert_allclose_53811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 69)
        assert_allclose_call_result_53831 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assert_allclose_53811, *[_test_cdotc_call_result_53825, result_add_53828, int_53829], **kwargs_53830)
        
        
        # Call to assert_allclose(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to _test_cdotu(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining the type of the subscript
        int_53835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'int')
        slice_53836 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 41), None, None, int_53835)
        # Getting the type of 'cx' (line 71)
        cx_53837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 41), 'cx', False)
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___53838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 41), cx_53837, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_53839 = invoke(stypy.reporting.localization.Localization(__file__, 71, 41), getitem___53838, slice_53836)
        
        
        # Obtaining the type of the subscript
        int_53840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 55), 'int')
        slice_53841 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 50), None, None, int_53840)
        # Getting the type of 'cy' (line 71)
        cy_53842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 50), 'cy', False)
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___53843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 50), cy_53842, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_53844 = invoke(stypy.reporting.localization.Localization(__file__, 71, 50), getitem___53843, slice_53841)
        
        # Processing the call keyword arguments (line 71)
        kwargs_53845 = {}
        # Getting the type of 'blas' (line 71)
        blas_53833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'blas', False)
        # Obtaining the member '_test_cdotu' of a type (line 71)
        _test_cdotu_53834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), blas_53833, '_test_cdotu')
        # Calling _test_cdotu(args, kwargs) (line 71)
        _test_cdotu_call_result_53846 = invoke(stypy.reporting.localization.Localization(__file__, 71, 24), _test_cdotu_53834, *[subscript_call_result_53839, subscript_call_result_53844], **kwargs_53845)
        
        float_53847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'float')
        complex_53848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 39), 'complex')
        # Applying the binary operator '+' (line 72)
        result_add_53849 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 24), '+', float_53847, complex_53848)
        
        int_53850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 55), 'int')
        # Processing the call keyword arguments (line 71)
        kwargs_53851 = {}
        # Getting the type of 'assert_allclose' (line 71)
        assert_allclose_53832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 71)
        assert_allclose_call_result_53852 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert_allclose_53832, *[_test_cdotu_call_result_53846, result_add_53849, int_53850], **kwargs_53851)
        
        
        # Call to assert_allclose(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to _test_scasum(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Obtaining the type of the subscript
        int_53856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 47), 'int')
        slice_53857 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 73, 42), None, None, int_53856)
        # Getting the type of 'cx' (line 73)
        cx_53858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'cx', False)
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___53859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 42), cx_53858, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_53860 = invoke(stypy.reporting.localization.Localization(__file__, 73, 42), getitem___53859, slice_53857)
        
        # Processing the call keyword arguments (line 73)
        kwargs_53861 = {}
        # Getting the type of 'blas' (line 73)
        blas_53854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'blas', False)
        # Obtaining the member '_test_scasum' of a type (line 73)
        _test_scasum_53855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 24), blas_53854, '_test_scasum')
        # Calling _test_scasum(args, kwargs) (line 73)
        _test_scasum_call_result_53862 = invoke(stypy.reporting.localization.Localization(__file__, 73, 24), _test_scasum_53855, *[subscript_call_result_53860], **kwargs_53861)
        
        float_53863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 52), 'float')
        int_53864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 57), 'int')
        # Processing the call keyword arguments (line 73)
        kwargs_53865 = {}
        # Getting the type of 'assert_allclose' (line 73)
        assert_allclose_53853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 73)
        assert_allclose_call_result_53866 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assert_allclose_53853, *[_test_scasum_call_result_53862, float_53863, int_53864], **kwargs_53865)
        
        
        # Call to assert_allclose(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to _test_scnrm2(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Obtaining the type of the subscript
        int_53870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 47), 'int')
        slice_53871 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 74, 42), None, None, int_53870)
        # Getting the type of 'cx' (line 74)
        cx_53872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 42), 'cx', False)
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___53873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 42), cx_53872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_53874 = invoke(stypy.reporting.localization.Localization(__file__, 74, 42), getitem___53873, slice_53871)
        
        # Processing the call keyword arguments (line 74)
        kwargs_53875 = {}
        # Getting the type of 'blas' (line 74)
        blas_53868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'blas', False)
        # Obtaining the member '_test_scnrm2' of a type (line 74)
        _test_scnrm2_53869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 24), blas_53868, '_test_scnrm2')
        # Calling _test_scnrm2(args, kwargs) (line 74)
        _test_scnrm2_call_result_53876 = invoke(stypy.reporting.localization.Localization(__file__, 74, 24), _test_scnrm2_53869, *[subscript_call_result_53874], **kwargs_53875)
        
        float_53877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 52), 'float')
        int_53878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 67), 'int')
        # Processing the call keyword arguments (line 74)
        kwargs_53879 = {}
        # Getting the type of 'assert_allclose' (line 74)
        assert_allclose_53867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 74)
        assert_allclose_call_result_53880 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assert_allclose_53867, *[_test_scnrm2_call_result_53876, float_53877, int_53878], **kwargs_53879)
        
        
        # ################# End of 'test_complex_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_complex_args' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_53881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_complex_args'
        return stypy_return_type_53881


    @norecursion
    def test_double_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_double_args'
        module_type_store = module_type_store.open_function_context('test_double_args', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_localization', localization)
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_function_name', 'TestWfuncPointers.test_double_args')
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWfuncPointers.test_double_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWfuncPointers.test_double_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_double_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_double_args(...)' code ##################

        
        # Assigning a Call to a Name (line 78):
        
        # Call to array(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_53884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        float_53885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_53884, float_53885)
        # Adding element type (line 78)
        int_53886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_53884, int_53886)
        # Adding element type (line 78)
        float_53887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_53884, float_53887)
        
        # Getting the type of 'np' (line 78)
        np_53888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 36), 'np', False)
        # Obtaining the member 'float64' of a type (line 78)
        float64_53889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 36), np_53888, 'float64')
        # Processing the call keyword arguments (line 78)
        kwargs_53890 = {}
        # Getting the type of 'np' (line 78)
        np_53882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 78)
        array_53883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), np_53882, 'array')
        # Calling array(args, kwargs) (line 78)
        array_call_result_53891 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), array_53883, *[list_53884, float64_53889], **kwargs_53890)
        
        # Assigning a type to the variable 'x' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'x', array_call_result_53891)
        
        # Assigning a Call to a Name (line 79):
        
        # Call to array(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_53894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        int_53895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 21), list_53894, int_53895)
        # Adding element type (line 79)
        int_53896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 21), list_53894, int_53896)
        # Adding element type (line 79)
        float_53897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 21), list_53894, float_53897)
        
        # Getting the type of 'np' (line 79)
        np_53898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'np', False)
        # Obtaining the member 'float64' of a type (line 79)
        float64_53899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 33), np_53898, 'float64')
        # Processing the call keyword arguments (line 79)
        kwargs_53900 = {}
        # Getting the type of 'np' (line 79)
        np_53892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 79)
        array_53893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), np_53892, 'array')
        # Calling array(args, kwargs) (line 79)
        array_call_result_53901 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), array_53893, *[list_53894, float64_53899], **kwargs_53900)
        
        # Assigning a type to the variable 'y' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'y', array_call_result_53901)
        
        # Call to assert_allclose(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Call to _test_dasum(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'x' (line 81)
        x_53905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 41), 'x', False)
        # Processing the call keyword arguments (line 81)
        kwargs_53906 = {}
        # Getting the type of 'blas' (line 81)
        blas_53903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'blas', False)
        # Obtaining the member '_test_dasum' of a type (line 81)
        _test_dasum_53904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 24), blas_53903, '_test_dasum')
        # Calling _test_dasum(args, kwargs) (line 81)
        _test_dasum_call_result_53907 = invoke(stypy.reporting.localization.Localization(__file__, 81, 24), _test_dasum_53904, *[x_53905], **kwargs_53906)
        
        float_53908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 45), 'float')
        int_53909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 50), 'int')
        # Processing the call keyword arguments (line 81)
        kwargs_53910 = {}
        # Getting the type of 'assert_allclose' (line 81)
        assert_allclose_53902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 81)
        assert_allclose_call_result_53911 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), assert_allclose_53902, *[_test_dasum_call_result_53907, float_53908, int_53909], **kwargs_53910)
        
        
        # Call to assert_allclose(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Call to _test_ddot(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'x' (line 82)
        x_53915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'x', False)
        # Getting the type of 'y' (line 82)
        y_53916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'y', False)
        # Processing the call keyword arguments (line 82)
        kwargs_53917 = {}
        # Getting the type of 'blas' (line 82)
        blas_53913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'blas', False)
        # Obtaining the member '_test_ddot' of a type (line 82)
        _test_ddot_53914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 24), blas_53913, '_test_ddot')
        # Calling _test_ddot(args, kwargs) (line 82)
        _test_ddot_call_result_53918 = invoke(stypy.reporting.localization.Localization(__file__, 82, 24), _test_ddot_53914, *[x_53915, y_53916], **kwargs_53917)
        
        float_53919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 47), 'float')
        int_53920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 53), 'int')
        # Processing the call keyword arguments (line 82)
        kwargs_53921 = {}
        # Getting the type of 'assert_allclose' (line 82)
        assert_allclose_53912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 82)
        assert_allclose_call_result_53922 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assert_allclose_53912, *[_test_ddot_call_result_53918, float_53919, int_53920], **kwargs_53921)
        
        
        # Call to assert_allclose(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to _test_dnrm2(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'x' (line 83)
        x_53926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'x', False)
        # Processing the call keyword arguments (line 83)
        kwargs_53927 = {}
        # Getting the type of 'blas' (line 83)
        blas_53924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'blas', False)
        # Obtaining the member '_test_dnrm2' of a type (line 83)
        _test_dnrm2_53925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 24), blas_53924, '_test_dnrm2')
        # Calling _test_dnrm2(args, kwargs) (line 83)
        _test_dnrm2_call_result_53928 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), _test_dnrm2_53925, *[x_53926], **kwargs_53927)
        
        float_53929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 45), 'float')
        int_53930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 60), 'int')
        # Processing the call keyword arguments (line 83)
        kwargs_53931 = {}
        # Getting the type of 'assert_allclose' (line 83)
        assert_allclose_53923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 83)
        assert_allclose_call_result_53932 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assert_allclose_53923, *[_test_dnrm2_call_result_53928, float_53929, int_53930], **kwargs_53931)
        
        
        # Call to assert_allclose(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to _test_dasum(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining the type of the subscript
        int_53936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 45), 'int')
        slice_53937 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 41), None, None, int_53936)
        # Getting the type of 'x' (line 85)
        x_53938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___53939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 41), x_53938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_53940 = invoke(stypy.reporting.localization.Localization(__file__, 85, 41), getitem___53939, slice_53937)
        
        # Processing the call keyword arguments (line 85)
        kwargs_53941 = {}
        # Getting the type of 'blas' (line 85)
        blas_53934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'blas', False)
        # Obtaining the member '_test_dasum' of a type (line 85)
        _test_dasum_53935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), blas_53934, '_test_dasum')
        # Calling _test_dasum(args, kwargs) (line 85)
        _test_dasum_call_result_53942 = invoke(stypy.reporting.localization.Localization(__file__, 85, 24), _test_dasum_53935, *[subscript_call_result_53940], **kwargs_53941)
        
        float_53943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 50), 'float')
        int_53944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 55), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_53945 = {}
        # Getting the type of 'assert_allclose' (line 85)
        assert_allclose_53933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 85)
        assert_allclose_call_result_53946 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assert_allclose_53933, *[_test_dasum_call_result_53942, float_53943, int_53944], **kwargs_53945)
        
        
        # Call to assert_allclose(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to _test_ddot(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining the type of the subscript
        int_53950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 44), 'int')
        slice_53951 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 86, 40), None, None, int_53950)
        # Getting the type of 'x' (line 86)
        x_53952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 40), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___53953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 40), x_53952, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_53954 = invoke(stypy.reporting.localization.Localization(__file__, 86, 40), getitem___53953, slice_53951)
        
        
        # Obtaining the type of the subscript
        int_53955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 52), 'int')
        slice_53956 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 86, 48), None, None, int_53955)
        # Getting the type of 'y' (line 86)
        y_53957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 48), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___53958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 48), y_53957, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_53959 = invoke(stypy.reporting.localization.Localization(__file__, 86, 48), getitem___53958, slice_53956)
        
        # Processing the call keyword arguments (line 86)
        kwargs_53960 = {}
        # Getting the type of 'blas' (line 86)
        blas_53948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'blas', False)
        # Obtaining the member '_test_ddot' of a type (line 86)
        _test_ddot_53949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), blas_53948, '_test_ddot')
        # Calling _test_ddot(args, kwargs) (line 86)
        _test_ddot_call_result_53961 = invoke(stypy.reporting.localization.Localization(__file__, 86, 24), _test_ddot_53949, *[subscript_call_result_53954, subscript_call_result_53959], **kwargs_53960)
        
        float_53962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 57), 'float')
        int_53963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 63), 'int')
        # Processing the call keyword arguments (line 86)
        kwargs_53964 = {}
        # Getting the type of 'assert_allclose' (line 86)
        assert_allclose_53947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 86)
        assert_allclose_call_result_53965 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), assert_allclose_53947, *[_test_ddot_call_result_53961, float_53962, int_53963], **kwargs_53964)
        
        
        # Call to assert_allclose(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to _test_dnrm2(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Obtaining the type of the subscript
        int_53969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 45), 'int')
        slice_53970 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 41), None, None, int_53969)
        # Getting the type of 'x' (line 87)
        x_53971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___53972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 41), x_53971, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_53973 = invoke(stypy.reporting.localization.Localization(__file__, 87, 41), getitem___53972, slice_53970)
        
        # Processing the call keyword arguments (line 87)
        kwargs_53974 = {}
        # Getting the type of 'blas' (line 87)
        blas_53967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'blas', False)
        # Obtaining the member '_test_dnrm2' of a type (line 87)
        _test_dnrm2_53968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), blas_53967, '_test_dnrm2')
        # Calling _test_dnrm2(args, kwargs) (line 87)
        _test_dnrm2_call_result_53975 = invoke(stypy.reporting.localization.Localization(__file__, 87, 24), _test_dnrm2_53968, *[subscript_call_result_53973], **kwargs_53974)
        
        float_53976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 50), 'float')
        int_53977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 64), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_53978 = {}
        # Getting the type of 'assert_allclose' (line 87)
        assert_allclose_53966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 87)
        assert_allclose_call_result_53979 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), assert_allclose_53966, *[_test_dnrm2_call_result_53975, float_53976, int_53977], **kwargs_53978)
        
        
        # Call to assert_equal(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to _test_idamax(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'x' (line 89)
        x_53983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'x', False)
        # Processing the call keyword arguments (line 89)
        kwargs_53984 = {}
        # Getting the type of 'blas' (line 89)
        blas_53981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'blas', False)
        # Obtaining the member '_test_idamax' of a type (line 89)
        _test_idamax_53982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 21), blas_53981, '_test_idamax')
        # Calling _test_idamax(args, kwargs) (line 89)
        _test_idamax_call_result_53985 = invoke(stypy.reporting.localization.Localization(__file__, 89, 21), _test_idamax_53982, *[x_53983], **kwargs_53984)
        
        int_53986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 43), 'int')
        # Processing the call keyword arguments (line 89)
        kwargs_53987 = {}
        # Getting the type of 'assert_equal' (line 89)
        assert_equal_53980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 89)
        assert_equal_call_result_53988 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert_equal_53980, *[_test_idamax_call_result_53985, int_53986], **kwargs_53987)
        
        
        # ################# End of 'test_double_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_double_args' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_53989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53989)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_double_args'
        return stypy_return_type_53989


    @norecursion
    def test_float_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_float_args'
        module_type_store = module_type_store.open_function_context('test_float_args', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_localization', localization)
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_function_name', 'TestWfuncPointers.test_float_args')
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWfuncPointers.test_float_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWfuncPointers.test_float_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_float_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_float_args(...)' code ##################

        
        # Assigning a Call to a Name (line 93):
        
        # Call to array(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_53992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        float_53993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), list_53992, float_53993)
        # Adding element type (line 93)
        int_53994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), list_53992, int_53994)
        # Adding element type (line 93)
        float_53995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), list_53992, float_53995)
        
        # Getting the type of 'np' (line 93)
        np_53996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 93)
        float32_53997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 36), np_53996, 'float32')
        # Processing the call keyword arguments (line 93)
        kwargs_53998 = {}
        # Getting the type of 'np' (line 93)
        np_53990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 93)
        array_53991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), np_53990, 'array')
        # Calling array(args, kwargs) (line 93)
        array_call_result_53999 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), array_53991, *[list_53992, float32_53997], **kwargs_53998)
        
        # Assigning a type to the variable 'x' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'x', array_call_result_53999)
        
        # Assigning a Call to a Name (line 94):
        
        # Call to array(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_54002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        int_54003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_54002, int_54003)
        # Adding element type (line 94)
        int_54004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_54002, int_54004)
        # Adding element type (line 94)
        float_54005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_54002, float_54005)
        
        # Getting the type of 'np' (line 94)
        np_54006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'np', False)
        # Obtaining the member 'float32' of a type (line 94)
        float32_54007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), np_54006, 'float32')
        # Processing the call keyword arguments (line 94)
        kwargs_54008 = {}
        # Getting the type of 'np' (line 94)
        np_54000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 94)
        array_54001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), np_54000, 'array')
        # Calling array(args, kwargs) (line 94)
        array_call_result_54009 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), array_54001, *[list_54002, float32_54007], **kwargs_54008)
        
        # Assigning a type to the variable 'y' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'y', array_call_result_54009)
        
        # Call to assert_equal(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to _test_isamax(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'x' (line 96)
        x_54013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 39), 'x', False)
        # Processing the call keyword arguments (line 96)
        kwargs_54014 = {}
        # Getting the type of 'blas' (line 96)
        blas_54011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'blas', False)
        # Obtaining the member '_test_isamax' of a type (line 96)
        _test_isamax_54012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 21), blas_54011, '_test_isamax')
        # Calling _test_isamax(args, kwargs) (line 96)
        _test_isamax_call_result_54015 = invoke(stypy.reporting.localization.Localization(__file__, 96, 21), _test_isamax_54012, *[x_54013], **kwargs_54014)
        
        int_54016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 43), 'int')
        # Processing the call keyword arguments (line 96)
        kwargs_54017 = {}
        # Getting the type of 'assert_equal' (line 96)
        assert_equal_54010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 96)
        assert_equal_call_result_54018 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assert_equal_54010, *[_test_isamax_call_result_54015, int_54016], **kwargs_54017)
        
        
        # Call to assert_allclose(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to _test_sasum(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x' (line 98)
        x_54022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'x', False)
        # Processing the call keyword arguments (line 98)
        kwargs_54023 = {}
        # Getting the type of 'blas' (line 98)
        blas_54020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'blas', False)
        # Obtaining the member '_test_sasum' of a type (line 98)
        _test_sasum_54021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), blas_54020, '_test_sasum')
        # Calling _test_sasum(args, kwargs) (line 98)
        _test_sasum_call_result_54024 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), _test_sasum_54021, *[x_54022], **kwargs_54023)
        
        float_54025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 45), 'float')
        int_54026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 50), 'int')
        # Processing the call keyword arguments (line 98)
        kwargs_54027 = {}
        # Getting the type of 'assert_allclose' (line 98)
        assert_allclose_54019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 98)
        assert_allclose_call_result_54028 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), assert_allclose_54019, *[_test_sasum_call_result_54024, float_54025, int_54026], **kwargs_54027)
        
        
        # Call to assert_allclose(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to _test_sdot(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'x' (line 99)
        x_54032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'x', False)
        # Getting the type of 'y' (line 99)
        y_54033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 43), 'y', False)
        # Processing the call keyword arguments (line 99)
        kwargs_54034 = {}
        # Getting the type of 'blas' (line 99)
        blas_54030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'blas', False)
        # Obtaining the member '_test_sdot' of a type (line 99)
        _test_sdot_54031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 24), blas_54030, '_test_sdot')
        # Calling _test_sdot(args, kwargs) (line 99)
        _test_sdot_call_result_54035 = invoke(stypy.reporting.localization.Localization(__file__, 99, 24), _test_sdot_54031, *[x_54032, y_54033], **kwargs_54034)
        
        float_54036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 47), 'float')
        int_54037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 53), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_54038 = {}
        # Getting the type of 'assert_allclose' (line 99)
        assert_allclose_54029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 99)
        assert_allclose_call_result_54039 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assert_allclose_54029, *[_test_sdot_call_result_54035, float_54036, int_54037], **kwargs_54038)
        
        
        # Call to assert_allclose(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to _test_snrm2(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'x' (line 100)
        x_54043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 41), 'x', False)
        # Processing the call keyword arguments (line 100)
        kwargs_54044 = {}
        # Getting the type of 'blas' (line 100)
        blas_54041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'blas', False)
        # Obtaining the member '_test_snrm2' of a type (line 100)
        _test_snrm2_54042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 24), blas_54041, '_test_snrm2')
        # Calling _test_snrm2(args, kwargs) (line 100)
        _test_snrm2_call_result_54045 = invoke(stypy.reporting.localization.Localization(__file__, 100, 24), _test_snrm2_54042, *[x_54043], **kwargs_54044)
        
        float_54046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 45), 'float')
        int_54047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 60), 'int')
        # Processing the call keyword arguments (line 100)
        kwargs_54048 = {}
        # Getting the type of 'assert_allclose' (line 100)
        assert_allclose_54040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 100)
        assert_allclose_call_result_54049 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), assert_allclose_54040, *[_test_snrm2_call_result_54045, float_54046, int_54047], **kwargs_54048)
        
        
        # Call to assert_allclose(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to _test_sasum(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Obtaining the type of the subscript
        int_54053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 45), 'int')
        slice_54054 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 102, 41), None, None, int_54053)
        # Getting the type of 'x' (line 102)
        x_54055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 41), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 102)
        getitem___54056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 41), x_54055, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 102)
        subscript_call_result_54057 = invoke(stypy.reporting.localization.Localization(__file__, 102, 41), getitem___54056, slice_54054)
        
        # Processing the call keyword arguments (line 102)
        kwargs_54058 = {}
        # Getting the type of 'blas' (line 102)
        blas_54051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'blas', False)
        # Obtaining the member '_test_sasum' of a type (line 102)
        _test_sasum_54052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), blas_54051, '_test_sasum')
        # Calling _test_sasum(args, kwargs) (line 102)
        _test_sasum_call_result_54059 = invoke(stypy.reporting.localization.Localization(__file__, 102, 24), _test_sasum_54052, *[subscript_call_result_54057], **kwargs_54058)
        
        float_54060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 50), 'float')
        int_54061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 55), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_54062 = {}
        # Getting the type of 'assert_allclose' (line 102)
        assert_allclose_54050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 102)
        assert_allclose_call_result_54063 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_allclose_54050, *[_test_sasum_call_result_54059, float_54060, int_54061], **kwargs_54062)
        
        
        # Call to assert_allclose(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to _test_sdot(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining the type of the subscript
        int_54067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 44), 'int')
        slice_54068 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 103, 40), None, None, int_54067)
        # Getting the type of 'x' (line 103)
        x_54069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___54070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 40), x_54069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_54071 = invoke(stypy.reporting.localization.Localization(__file__, 103, 40), getitem___54070, slice_54068)
        
        
        # Obtaining the type of the subscript
        int_54072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 52), 'int')
        slice_54073 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 103, 48), None, None, int_54072)
        # Getting the type of 'y' (line 103)
        y_54074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 48), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___54075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 48), y_54074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_54076 = invoke(stypy.reporting.localization.Localization(__file__, 103, 48), getitem___54075, slice_54073)
        
        # Processing the call keyword arguments (line 103)
        kwargs_54077 = {}
        # Getting the type of 'blas' (line 103)
        blas_54065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'blas', False)
        # Obtaining the member '_test_sdot' of a type (line 103)
        _test_sdot_54066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 24), blas_54065, '_test_sdot')
        # Calling _test_sdot(args, kwargs) (line 103)
        _test_sdot_call_result_54078 = invoke(stypy.reporting.localization.Localization(__file__, 103, 24), _test_sdot_54066, *[subscript_call_result_54071, subscript_call_result_54076], **kwargs_54077)
        
        float_54079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 57), 'float')
        int_54080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 63), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_54081 = {}
        # Getting the type of 'assert_allclose' (line 103)
        assert_allclose_54064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 103)
        assert_allclose_call_result_54082 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assert_allclose_54064, *[_test_sdot_call_result_54078, float_54079, int_54080], **kwargs_54081)
        
        
        # Call to assert_allclose(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to _test_snrm2(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining the type of the subscript
        int_54086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 45), 'int')
        slice_54087 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 41), None, None, int_54086)
        # Getting the type of 'x' (line 104)
        x_54088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 41), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___54089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 41), x_54088, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_54090 = invoke(stypy.reporting.localization.Localization(__file__, 104, 41), getitem___54089, slice_54087)
        
        # Processing the call keyword arguments (line 104)
        kwargs_54091 = {}
        # Getting the type of 'blas' (line 104)
        blas_54084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'blas', False)
        # Obtaining the member '_test_snrm2' of a type (line 104)
        _test_snrm2_54085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 24), blas_54084, '_test_snrm2')
        # Calling _test_snrm2(args, kwargs) (line 104)
        _test_snrm2_call_result_54092 = invoke(stypy.reporting.localization.Localization(__file__, 104, 24), _test_snrm2_54085, *[subscript_call_result_54090], **kwargs_54091)
        
        float_54093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 50), 'float')
        int_54094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 64), 'int')
        # Processing the call keyword arguments (line 104)
        kwargs_54095 = {}
        # Getting the type of 'assert_allclose' (line 104)
        assert_allclose_54083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 104)
        assert_allclose_call_result_54096 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assert_allclose_54083, *[_test_snrm2_call_result_54092, float_54093, int_54094], **kwargs_54095)
        
        
        # ################# End of 'test_float_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_float_args' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_54097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_float_args'
        return stypy_return_type_54097


    @norecursion
    def test_double_complex_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_double_complex_args'
        module_type_store = module_type_store.open_function_context('test_double_complex_args', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_localization', localization)
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_function_name', 'TestWfuncPointers.test_double_complex_args')
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWfuncPointers.test_double_complex_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWfuncPointers.test_double_complex_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_double_complex_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_double_complex_args(...)' code ##################

        
        # Assigning a Call to a Name (line 108):
        
        # Call to array(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_54100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        float_54101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'float')
        complex_54102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 28), 'complex')
        # Applying the binary operator '+' (line 108)
        result_add_54103 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 23), '+', float_54101, complex_54102)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_54100, result_add_54103)
        # Adding element type (line 108)
        float_54104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 33), 'float')
        complex_54105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'complex')
        # Applying the binary operator '-' (line 108)
        result_sub_54106 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 33), '-', float_54104, complex_54105)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_54100, result_sub_54106)
        # Adding element type (line 108)
        float_54107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 46), 'float')
        complex_54108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 52), 'complex')
        # Applying the binary operator '-' (line 108)
        result_sub_54109 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 46), '-', float_54107, complex_54108)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_54100, result_sub_54109)
        
        # Getting the type of 'np' (line 108)
        np_54110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 58), 'np', False)
        # Obtaining the member 'complex128' of a type (line 108)
        complex128_54111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 58), np_54110, 'complex128')
        # Processing the call keyword arguments (line 108)
        kwargs_54112 = {}
        # Getting the type of 'np' (line 108)
        np_54098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 108)
        array_54099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), np_54098, 'array')
        # Calling array(args, kwargs) (line 108)
        array_call_result_54113 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), array_54099, *[list_54100, complex128_54111], **kwargs_54112)
        
        # Assigning a type to the variable 'cx' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'cx', array_call_result_54113)
        
        # Assigning a Call to a Name (line 109):
        
        # Call to array(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_54116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        float_54117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'float')
        complex_54118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'complex')
        # Applying the binary operator '+' (line 109)
        result_add_54119 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 23), '+', float_54117, complex_54118)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_54116, result_add_54119)
        # Adding element type (line 109)
        float_54120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'float')
        complex_54121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 42), 'complex')
        # Applying the binary operator '-' (line 109)
        result_sub_54122 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 35), '-', float_54120, complex_54121)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_54116, result_sub_54122)
        # Adding element type (line 109)
        float_54123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 49), 'float')
        complex_54124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 55), 'complex')
        # Applying the binary operator '+' (line 109)
        result_add_54125 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 49), '+', float_54123, complex_54124)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_54116, result_add_54125)
        
        # Getting the type of 'np' (line 109)
        np_54126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 61), 'np', False)
        # Obtaining the member 'complex128' of a type (line 109)
        complex128_54127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 61), np_54126, 'complex128')
        # Processing the call keyword arguments (line 109)
        kwargs_54128 = {}
        # Getting the type of 'np' (line 109)
        np_54114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 109)
        array_54115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), np_54114, 'array')
        # Calling array(args, kwargs) (line 109)
        array_call_result_54129 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), array_54115, *[list_54116, complex128_54127], **kwargs_54128)
        
        # Assigning a type to the variable 'cy' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'cy', array_call_result_54129)
        
        # Call to assert_equal(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to _test_izamax(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'cx' (line 111)
        cx_54133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 39), 'cx', False)
        # Processing the call keyword arguments (line 111)
        kwargs_54134 = {}
        # Getting the type of 'blas' (line 111)
        blas_54131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'blas', False)
        # Obtaining the member '_test_izamax' of a type (line 111)
        _test_izamax_54132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 21), blas_54131, '_test_izamax')
        # Calling _test_izamax(args, kwargs) (line 111)
        _test_izamax_call_result_54135 = invoke(stypy.reporting.localization.Localization(__file__, 111, 21), _test_izamax_54132, *[cx_54133], **kwargs_54134)
        
        int_54136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 44), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_54137 = {}
        # Getting the type of 'assert_equal' (line 111)
        assert_equal_54130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 111)
        assert_equal_call_result_54138 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assert_equal_54130, *[_test_izamax_call_result_54135, int_54136], **kwargs_54137)
        
        
        # Call to assert_allclose(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to _test_zdotc(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'cx' (line 113)
        cx_54142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'cx', False)
        # Getting the type of 'cy' (line 113)
        cy_54143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 45), 'cy', False)
        # Processing the call keyword arguments (line 113)
        kwargs_54144 = {}
        # Getting the type of 'blas' (line 113)
        blas_54140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'blas', False)
        # Obtaining the member '_test_zdotc' of a type (line 113)
        _test_zdotc_54141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), blas_54140, '_test_zdotc')
        # Calling _test_zdotc(args, kwargs) (line 113)
        _test_zdotc_call_result_54145 = invoke(stypy.reporting.localization.Localization(__file__, 113, 24), _test_zdotc_54141, *[cx_54142, cy_54143], **kwargs_54144)
        
        float_54146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 50), 'float')
        complex_54147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 61), 'complex')
        # Applying the binary operator '+' (line 113)
        result_add_54148 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 50), '+', float_54146, complex_54147)
        
        int_54149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 73), 'int')
        # Processing the call keyword arguments (line 113)
        kwargs_54150 = {}
        # Getting the type of 'assert_allclose' (line 113)
        assert_allclose_54139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 113)
        assert_allclose_call_result_54151 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_allclose_54139, *[_test_zdotc_call_result_54145, result_add_54148, int_54149], **kwargs_54150)
        
        
        # Call to assert_allclose(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to _test_zdotu(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'cx' (line 114)
        cx_54155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 41), 'cx', False)
        # Getting the type of 'cy' (line 114)
        cy_54156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 45), 'cy', False)
        # Processing the call keyword arguments (line 114)
        kwargs_54157 = {}
        # Getting the type of 'blas' (line 114)
        blas_54153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'blas', False)
        # Obtaining the member '_test_zdotu' of a type (line 114)
        _test_zdotu_54154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 24), blas_54153, '_test_zdotu')
        # Calling _test_zdotu(args, kwargs) (line 114)
        _test_zdotu_call_result_54158 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), _test_zdotu_54154, *[cx_54155, cy_54156], **kwargs_54157)
        
        float_54159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 50), 'float')
        complex_54160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 60), 'complex')
        # Applying the binary operator '+' (line 114)
        result_add_54161 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 50), '+', float_54159, complex_54160)
        
        int_54162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 72), 'int')
        # Processing the call keyword arguments (line 114)
        kwargs_54163 = {}
        # Getting the type of 'assert_allclose' (line 114)
        assert_allclose_54152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 114)
        assert_allclose_call_result_54164 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assert_allclose_54152, *[_test_zdotu_call_result_54158, result_add_54161, int_54162], **kwargs_54163)
        
        
        # Call to assert_allclose(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to _test_zdotc(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining the type of the subscript
        int_54168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 46), 'int')
        slice_54169 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 116, 41), None, None, int_54168)
        # Getting the type of 'cx' (line 116)
        cx_54170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 41), 'cx', False)
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___54171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 41), cx_54170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_54172 = invoke(stypy.reporting.localization.Localization(__file__, 116, 41), getitem___54171, slice_54169)
        
        
        # Obtaining the type of the subscript
        int_54173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 55), 'int')
        slice_54174 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 116, 50), None, None, int_54173)
        # Getting the type of 'cy' (line 116)
        cy_54175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 50), 'cy', False)
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___54176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 50), cy_54175, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_54177 = invoke(stypy.reporting.localization.Localization(__file__, 116, 50), getitem___54176, slice_54174)
        
        # Processing the call keyword arguments (line 116)
        kwargs_54178 = {}
        # Getting the type of 'blas' (line 116)
        blas_54166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'blas', False)
        # Obtaining the member '_test_zdotc' of a type (line 116)
        _test_zdotc_54167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), blas_54166, '_test_zdotc')
        # Calling _test_zdotc(args, kwargs) (line 116)
        _test_zdotc_call_result_54179 = invoke(stypy.reporting.localization.Localization(__file__, 116, 24), _test_zdotc_54167, *[subscript_call_result_54172, subscript_call_result_54177], **kwargs_54178)
        
        float_54180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 24), 'float')
        complex_54181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'complex')
        # Applying the binary operator '+' (line 117)
        result_add_54182 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 24), '+', float_54180, complex_54181)
        
        int_54183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 42), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_54184 = {}
        # Getting the type of 'assert_allclose' (line 116)
        assert_allclose_54165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 116)
        assert_allclose_call_result_54185 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert_allclose_54165, *[_test_zdotc_call_result_54179, result_add_54182, int_54183], **kwargs_54184)
        
        
        # Call to assert_allclose(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to _test_zdotu(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Obtaining the type of the subscript
        int_54189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 46), 'int')
        slice_54190 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 118, 41), None, None, int_54189)
        # Getting the type of 'cx' (line 118)
        cx_54191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 41), 'cx', False)
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___54192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 41), cx_54191, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_54193 = invoke(stypy.reporting.localization.Localization(__file__, 118, 41), getitem___54192, slice_54190)
        
        
        # Obtaining the type of the subscript
        int_54194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 55), 'int')
        slice_54195 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 118, 50), None, None, int_54194)
        # Getting the type of 'cy' (line 118)
        cy_54196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 50), 'cy', False)
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___54197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 50), cy_54196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_54198 = invoke(stypy.reporting.localization.Localization(__file__, 118, 50), getitem___54197, slice_54195)
        
        # Processing the call keyword arguments (line 118)
        kwargs_54199 = {}
        # Getting the type of 'blas' (line 118)
        blas_54187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'blas', False)
        # Obtaining the member '_test_zdotu' of a type (line 118)
        _test_zdotu_54188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 24), blas_54187, '_test_zdotu')
        # Calling _test_zdotu(args, kwargs) (line 118)
        _test_zdotu_call_result_54200 = invoke(stypy.reporting.localization.Localization(__file__, 118, 24), _test_zdotu_54188, *[subscript_call_result_54193, subscript_call_result_54198], **kwargs_54199)
        
        float_54201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 24), 'float')
        complex_54202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'complex')
        # Applying the binary operator '+' (line 119)
        result_add_54203 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 24), '+', float_54201, complex_54202)
        
        int_54204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 41), 'int')
        # Processing the call keyword arguments (line 118)
        kwargs_54205 = {}
        # Getting the type of 'assert_allclose' (line 118)
        assert_allclose_54186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 118)
        assert_allclose_call_result_54206 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), assert_allclose_54186, *[_test_zdotu_call_result_54200, result_add_54203, int_54204], **kwargs_54205)
        
        
        # ################# End of 'test_double_complex_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_double_complex_args' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_54207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54207)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_double_complex_args'
        return stypy_return_type_54207


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 49, 0, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWfuncPointers.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestWfuncPointers' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'TestWfuncPointers', TestWfuncPointers)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
