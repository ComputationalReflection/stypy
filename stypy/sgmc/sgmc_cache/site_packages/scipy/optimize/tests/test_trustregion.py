
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit tests for trust-region optimization routines.
3: 
4: To run it in its simplest form::
5:   nosetests test_optimize.py
6: 
7: '''
8: from __future__ import division, print_function, absolute_import
9: 
10: import numpy as np
11: from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
12:                             rosen_hess_prod)
13: from numpy.testing import assert_, assert_equal, assert_allclose
14: 
15: 
16: class Accumulator:
17:     ''' This is for testing callbacks.'''
18:     def __init__(self):
19:         self.count = 0
20:         self.accum = None
21: 
22:     def __call__(self, x):
23:         self.count += 1
24:         if self.accum is None:
25:             self.accum = np.array(x)
26:         else:
27:             self.accum += x
28: 
29: 
30: class TestTrustRegionSolvers(object):
31: 
32:     def setup_method(self):
33:         self.x_opt = [1.0, 1.0]
34:         self.easy_guess = [2.0, 2.0]
35:         self.hard_guess = [-1.2, 1.0]
36: 
37:     def test_dogleg_accuracy(self):
38:         # test the accuracy and the return_all option
39:         x0 = self.hard_guess
40:         r = minimize(rosen, x0, jac=rosen_der, hess=rosen_hess, tol=1e-8,
41:                      method='dogleg', options={'return_all': True},)
42:         assert_allclose(x0, r['allvecs'][0])
43:         assert_allclose(r['x'], r['allvecs'][-1])
44:         assert_allclose(r['x'], self.x_opt)
45: 
46:     def test_dogleg_callback(self):
47:         # test the callback mechanism and the maxiter and return_all options
48:         accumulator = Accumulator()
49:         maxiter = 5
50:         r = minimize(rosen, self.hard_guess, jac=rosen_der, hess=rosen_hess,
51:                      callback=accumulator, method='dogleg',
52:                      options={'return_all': True, 'maxiter': maxiter},)
53:         assert_equal(accumulator.count, maxiter)
54:         assert_equal(len(r['allvecs']), maxiter+1)
55:         assert_allclose(r['x'], r['allvecs'][-1])
56:         assert_allclose(sum(r['allvecs'][1:]), accumulator.accum)
57: 
58:     def test_solver_concordance(self):
59:         # Assert that dogleg uses fewer iterations than ncg on the Rosenbrock
60:         # test function, although this does not necessarily mean
61:         # that dogleg is faster or better than ncg even for this function
62:         # and especially not for other test functions.
63:         f = rosen
64:         g = rosen_der
65:         h = rosen_hess
66:         for x0 in (self.easy_guess, self.hard_guess):
67:             r_dogleg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
68:                                 method='dogleg', options={'return_all': True})
69:             r_trust_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
70:                                    method='trust-ncg',
71:                                    options={'return_all': True})
72:             r_trust_krylov = minimize(f, x0, jac=g, hess=h, tol=1e-8,
73:                                    method='trust-krylov',
74:                                    options={'return_all': True})
75:             r_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
76:                              method='newton-cg', options={'return_all': True})
77:             r_iterative = minimize(f, x0, jac=g, hess=h, tol=1e-8,
78:                                    method='trust-exact',
79:                                    options={'return_all': True})
80:             assert_allclose(self.x_opt, r_dogleg['x'])
81:             assert_allclose(self.x_opt, r_trust_ncg['x'])
82:             assert_allclose(self.x_opt, r_trust_krylov['x'])
83:             assert_allclose(self.x_opt, r_ncg['x'])
84:             assert_allclose(self.x_opt, r_iterative['x'])
85:             assert_(len(r_dogleg['allvecs']) < len(r_ncg['allvecs']))
86: 
87:     def test_trust_ncg_hessp(self):
88:         for x0 in (self.easy_guess, self.hard_guess):
89:             r = minimize(rosen, x0, jac=rosen_der, hessp=rosen_hess_prod,
90:                          tol=1e-8, method='trust-ncg')
91:             assert_allclose(self.x_opt, r['x'])
92: 
93: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_233895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nUnit tests for trust-region optimization routines.\n\nTo run it in its simplest form::\n  nosetests test_optimize.py\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_233896 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_233896) is not StypyTypeError):

    if (import_233896 != 'pyd_module'):
        __import__(import_233896)
        sys_modules_233897 = sys.modules[import_233896]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_233897.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_233896)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize import minimize, rosen, rosen_der, rosen_hess, rosen_hess_prod' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_233898 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize')

if (type(import_233898) is not StypyTypeError):

    if (import_233898 != 'pyd_module'):
        __import__(import_233898)
        sys_modules_233899 = sys.modules[import_233898]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', sys_modules_233899.module_type_store, module_type_store, ['minimize', 'rosen', 'rosen_der', 'rosen_hess', 'rosen_hess_prod'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_233899, sys_modules_233899.module_type_store, module_type_store)
    else:
        from scipy.optimize import minimize, rosen, rosen_der, rosen_hess, rosen_hess_prod

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', None, module_type_store, ['minimize', 'rosen', 'rosen_der', 'rosen_hess', 'rosen_hess_prod'], [minimize, rosen, rosen_der, rosen_hess, rosen_hess_prod])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', import_233898)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.testing import assert_, assert_equal, assert_allclose' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_233900 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing')

if (type(import_233900) is not StypyTypeError):

    if (import_233900 != 'pyd_module'):
        __import__(import_233900)
        sys_modules_233901 = sys.modules[import_233900]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', sys_modules_233901.module_type_store, module_type_store, ['assert_', 'assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_233901, sys_modules_233901.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal', 'assert_allclose'], [assert_, assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', import_233900)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'Accumulator' class

class Accumulator:
    str_233902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', ' This is for testing callbacks.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Accumulator.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 19):
        int_233903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'int')
        # Getting the type of 'self' (line 19)
        self_233904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member 'count' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_233904, 'count', int_233903)
        
        # Assigning a Name to a Attribute (line 20):
        # Getting the type of 'None' (line 20)
        None_233905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'None')
        # Getting the type of 'self' (line 20)
        self_233906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'accum' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_233906, 'accum', None_233905)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Accumulator.__call__.__dict__.__setitem__('stypy_localization', localization)
        Accumulator.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Accumulator.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Accumulator.__call__.__dict__.__setitem__('stypy_function_name', 'Accumulator.__call__')
        Accumulator.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        Accumulator.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Accumulator.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Accumulator.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Accumulator.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Accumulator.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Accumulator.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Accumulator.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Getting the type of 'self' (line 23)
        self_233907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Obtaining the member 'count' of a type (line 23)
        count_233908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_233907, 'count')
        int_233909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'int')
        # Applying the binary operator '+=' (line 23)
        result_iadd_233910 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 8), '+=', count_233908, int_233909)
        # Getting the type of 'self' (line 23)
        self_233911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'count' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_233911, 'count', result_iadd_233910)
        
        
        # Type idiom detected: calculating its left and rigth part (line 24)
        # Getting the type of 'self' (line 24)
        self_233912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'self')
        # Obtaining the member 'accum' of a type (line 24)
        accum_233913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 11), self_233912, 'accum')
        # Getting the type of 'None' (line 24)
        None_233914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'None')
        
        (may_be_233915, more_types_in_union_233916) = may_be_none(accum_233913, None_233914)

        if may_be_233915:

            if more_types_in_union_233916:
                # Runtime conditional SSA (line 24)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 25):
            
            # Call to array(...): (line 25)
            # Processing the call arguments (line 25)
            # Getting the type of 'x' (line 25)
            x_233919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'x', False)
            # Processing the call keyword arguments (line 25)
            kwargs_233920 = {}
            # Getting the type of 'np' (line 25)
            np_233917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'np', False)
            # Obtaining the member 'array' of a type (line 25)
            array_233918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), np_233917, 'array')
            # Calling array(args, kwargs) (line 25)
            array_call_result_233921 = invoke(stypy.reporting.localization.Localization(__file__, 25, 25), array_233918, *[x_233919], **kwargs_233920)
            
            # Getting the type of 'self' (line 25)
            self_233922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'self')
            # Setting the type of the member 'accum' of a type (line 25)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), self_233922, 'accum', array_call_result_233921)

            if more_types_in_union_233916:
                # Runtime conditional SSA for else branch (line 24)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_233915) or more_types_in_union_233916):
            
            # Getting the type of 'self' (line 27)
            self_233923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self')
            # Obtaining the member 'accum' of a type (line 27)
            accum_233924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_233923, 'accum')
            # Getting the type of 'x' (line 27)
            x_233925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'x')
            # Applying the binary operator '+=' (line 27)
            result_iadd_233926 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 12), '+=', accum_233924, x_233925)
            # Getting the type of 'self' (line 27)
            self_233927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self')
            # Setting the type of the member 'accum' of a type (line 27)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_233927, 'accum', result_iadd_233926)
            

            if (may_be_233915 and more_types_in_union_233916):
                # SSA join for if statement (line 24)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_233928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_233928


# Assigning a type to the variable 'Accumulator' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'Accumulator', Accumulator)
# Declaration of the 'TestTrustRegionSolvers' class

class TestTrustRegionSolvers(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_function_name', 'TestTrustRegionSolvers.setup_method')
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTrustRegionSolvers.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTrustRegionSolvers.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 33):
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_233929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        float_233930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_233929, float_233930)
        # Adding element type (line 33)
        float_233931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_233929, float_233931)
        
        # Getting the type of 'self' (line 33)
        self_233932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'x_opt' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_233932, 'x_opt', list_233929)
        
        # Assigning a List to a Attribute (line 34):
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_233933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        float_233934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), list_233933, float_233934)
        # Adding element type (line 34)
        float_233935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), list_233933, float_233935)
        
        # Getting the type of 'self' (line 34)
        self_233936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'easy_guess' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_233936, 'easy_guess', list_233933)
        
        # Assigning a List to a Attribute (line 35):
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_233937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        float_233938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 26), list_233937, float_233938)
        # Adding element type (line 35)
        float_233939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 26), list_233937, float_233939)
        
        # Getting the type of 'self' (line 35)
        self_233940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'hard_guess' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_233940, 'hard_guess', list_233937)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_233941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233941)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_233941


    @norecursion
    def test_dogleg_accuracy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dogleg_accuracy'
        module_type_store = module_type_store.open_function_context('test_dogleg_accuracy', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_localization', localization)
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_function_name', 'TestTrustRegionSolvers.test_dogleg_accuracy')
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_param_names_list', [])
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTrustRegionSolvers.test_dogleg_accuracy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTrustRegionSolvers.test_dogleg_accuracy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dogleg_accuracy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dogleg_accuracy(...)' code ##################

        
        # Assigning a Attribute to a Name (line 39):
        # Getting the type of 'self' (line 39)
        self_233942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'self')
        # Obtaining the member 'hard_guess' of a type (line 39)
        hard_guess_233943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 13), self_233942, 'hard_guess')
        # Assigning a type to the variable 'x0' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'x0', hard_guess_233943)
        
        # Assigning a Call to a Name (line 40):
        
        # Call to minimize(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'rosen' (line 40)
        rosen_233945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'rosen', False)
        # Getting the type of 'x0' (line 40)
        x0_233946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'x0', False)
        # Processing the call keyword arguments (line 40)
        # Getting the type of 'rosen_der' (line 40)
        rosen_der_233947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'rosen_der', False)
        keyword_233948 = rosen_der_233947
        # Getting the type of 'rosen_hess' (line 40)
        rosen_hess_233949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 52), 'rosen_hess', False)
        keyword_233950 = rosen_hess_233949
        float_233951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 68), 'float')
        keyword_233952 = float_233951
        str_233953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'str', 'dogleg')
        keyword_233954 = str_233953
        
        # Obtaining an instance of the builtin type 'dict' (line 41)
        dict_233955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 41)
        # Adding element type (key, value) (line 41)
        str_233956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 47), 'str', 'return_all')
        # Getting the type of 'True' (line 41)
        True_233957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 61), 'True', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 46), dict_233955, (str_233956, True_233957))
        
        keyword_233958 = dict_233955
        kwargs_233959 = {'hess': keyword_233950, 'options': keyword_233958, 'jac': keyword_233948, 'tol': keyword_233952, 'method': keyword_233954}
        # Getting the type of 'minimize' (line 40)
        minimize_233944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'minimize', False)
        # Calling minimize(args, kwargs) (line 40)
        minimize_call_result_233960 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), minimize_233944, *[rosen_233945, x0_233946], **kwargs_233959)
        
        # Assigning a type to the variable 'r' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'r', minimize_call_result_233960)
        
        # Call to assert_allclose(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'x0' (line 42)
        x0_233962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'x0', False)
        
        # Obtaining the type of the subscript
        int_233963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 41), 'int')
        
        # Obtaining the type of the subscript
        str_233964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 30), 'str', 'allvecs')
        # Getting the type of 'r' (line 42)
        r_233965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___233966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 28), r_233965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_233967 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), getitem___233966, str_233964)
        
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___233968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 28), subscript_call_result_233967, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_233969 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), getitem___233968, int_233963)
        
        # Processing the call keyword arguments (line 42)
        kwargs_233970 = {}
        # Getting the type of 'assert_allclose' (line 42)
        assert_allclose_233961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 42)
        assert_allclose_call_result_233971 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_allclose_233961, *[x0_233962, subscript_call_result_233969], **kwargs_233970)
        
        
        # Call to assert_allclose(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Obtaining the type of the subscript
        str_233973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'str', 'x')
        # Getting the type of 'r' (line 43)
        r_233974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___233975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), r_233974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_233976 = invoke(stypy.reporting.localization.Localization(__file__, 43, 24), getitem___233975, str_233973)
        
        
        # Obtaining the type of the subscript
        int_233977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 45), 'int')
        
        # Obtaining the type of the subscript
        str_233978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'str', 'allvecs')
        # Getting the type of 'r' (line 43)
        r_233979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___233980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), r_233979, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_233981 = invoke(stypy.reporting.localization.Localization(__file__, 43, 32), getitem___233980, str_233978)
        
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___233982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), subscript_call_result_233981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_233983 = invoke(stypy.reporting.localization.Localization(__file__, 43, 32), getitem___233982, int_233977)
        
        # Processing the call keyword arguments (line 43)
        kwargs_233984 = {}
        # Getting the type of 'assert_allclose' (line 43)
        assert_allclose_233972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 43)
        assert_allclose_call_result_233985 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), assert_allclose_233972, *[subscript_call_result_233976, subscript_call_result_233983], **kwargs_233984)
        
        
        # Call to assert_allclose(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Obtaining the type of the subscript
        str_233987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'str', 'x')
        # Getting the type of 'r' (line 44)
        r_233988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___233989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), r_233988, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_233990 = invoke(stypy.reporting.localization.Localization(__file__, 44, 24), getitem___233989, str_233987)
        
        # Getting the type of 'self' (line 44)
        self_233991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 32), 'self', False)
        # Obtaining the member 'x_opt' of a type (line 44)
        x_opt_233992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 32), self_233991, 'x_opt')
        # Processing the call keyword arguments (line 44)
        kwargs_233993 = {}
        # Getting the type of 'assert_allclose' (line 44)
        assert_allclose_233986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 44)
        assert_allclose_call_result_233994 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), assert_allclose_233986, *[subscript_call_result_233990, x_opt_233992], **kwargs_233993)
        
        
        # ################# End of 'test_dogleg_accuracy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dogleg_accuracy' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_233995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233995)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dogleg_accuracy'
        return stypy_return_type_233995


    @norecursion
    def test_dogleg_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dogleg_callback'
        module_type_store = module_type_store.open_function_context('test_dogleg_callback', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_localization', localization)
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_function_name', 'TestTrustRegionSolvers.test_dogleg_callback')
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_param_names_list', [])
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTrustRegionSolvers.test_dogleg_callback.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTrustRegionSolvers.test_dogleg_callback', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dogleg_callback', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dogleg_callback(...)' code ##################

        
        # Assigning a Call to a Name (line 48):
        
        # Call to Accumulator(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_233997 = {}
        # Getting the type of 'Accumulator' (line 48)
        Accumulator_233996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'Accumulator', False)
        # Calling Accumulator(args, kwargs) (line 48)
        Accumulator_call_result_233998 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), Accumulator_233996, *[], **kwargs_233997)
        
        # Assigning a type to the variable 'accumulator' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'accumulator', Accumulator_call_result_233998)
        
        # Assigning a Num to a Name (line 49):
        int_233999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'int')
        # Assigning a type to the variable 'maxiter' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'maxiter', int_233999)
        
        # Assigning a Call to a Name (line 50):
        
        # Call to minimize(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'rosen' (line 50)
        rosen_234001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'rosen', False)
        # Getting the type of 'self' (line 50)
        self_234002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'self', False)
        # Obtaining the member 'hard_guess' of a type (line 50)
        hard_guess_234003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 28), self_234002, 'hard_guess')
        # Processing the call keyword arguments (line 50)
        # Getting the type of 'rosen_der' (line 50)
        rosen_der_234004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 49), 'rosen_der', False)
        keyword_234005 = rosen_der_234004
        # Getting the type of 'rosen_hess' (line 50)
        rosen_hess_234006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 65), 'rosen_hess', False)
        keyword_234007 = rosen_hess_234006
        # Getting the type of 'accumulator' (line 51)
        accumulator_234008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'accumulator', False)
        keyword_234009 = accumulator_234008
        str_234010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 50), 'str', 'dogleg')
        keyword_234011 = str_234010
        
        # Obtaining an instance of the builtin type 'dict' (line 52)
        dict_234012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 52)
        # Adding element type (key, value) (line 52)
        str_234013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'str', 'return_all')
        # Getting the type of 'True' (line 52)
        True_234014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 44), 'True', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 29), dict_234012, (str_234013, True_234014))
        # Adding element type (key, value) (line 52)
        str_234015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 50), 'str', 'maxiter')
        # Getting the type of 'maxiter' (line 52)
        maxiter_234016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 61), 'maxiter', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 29), dict_234012, (str_234015, maxiter_234016))
        
        keyword_234017 = dict_234012
        kwargs_234018 = {'callback': keyword_234009, 'hess': keyword_234007, 'options': keyword_234017, 'jac': keyword_234005, 'method': keyword_234011}
        # Getting the type of 'minimize' (line 50)
        minimize_234000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'minimize', False)
        # Calling minimize(args, kwargs) (line 50)
        minimize_call_result_234019 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), minimize_234000, *[rosen_234001, hard_guess_234003], **kwargs_234018)
        
        # Assigning a type to the variable 'r' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'r', minimize_call_result_234019)
        
        # Call to assert_equal(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'accumulator' (line 53)
        accumulator_234021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'accumulator', False)
        # Obtaining the member 'count' of a type (line 53)
        count_234022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 21), accumulator_234021, 'count')
        # Getting the type of 'maxiter' (line 53)
        maxiter_234023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 40), 'maxiter', False)
        # Processing the call keyword arguments (line 53)
        kwargs_234024 = {}
        # Getting the type of 'assert_equal' (line 53)
        assert_equal_234020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 53)
        assert_equal_call_result_234025 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert_equal_234020, *[count_234022, maxiter_234023], **kwargs_234024)
        
        
        # Call to assert_equal(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to len(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining the type of the subscript
        str_234028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'str', 'allvecs')
        # Getting the type of 'r' (line 54)
        r_234029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___234030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), r_234029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_234031 = invoke(stypy.reporting.localization.Localization(__file__, 54, 25), getitem___234030, str_234028)
        
        # Processing the call keyword arguments (line 54)
        kwargs_234032 = {}
        # Getting the type of 'len' (line 54)
        len_234027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'len', False)
        # Calling len(args, kwargs) (line 54)
        len_call_result_234033 = invoke(stypy.reporting.localization.Localization(__file__, 54, 21), len_234027, *[subscript_call_result_234031], **kwargs_234032)
        
        # Getting the type of 'maxiter' (line 54)
        maxiter_234034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'maxiter', False)
        int_234035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 48), 'int')
        # Applying the binary operator '+' (line 54)
        result_add_234036 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 40), '+', maxiter_234034, int_234035)
        
        # Processing the call keyword arguments (line 54)
        kwargs_234037 = {}
        # Getting the type of 'assert_equal' (line 54)
        assert_equal_234026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 54)
        assert_equal_call_result_234038 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert_equal_234026, *[len_call_result_234033, result_add_234036], **kwargs_234037)
        
        
        # Call to assert_allclose(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining the type of the subscript
        str_234040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'str', 'x')
        # Getting the type of 'r' (line 55)
        r_234041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___234042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), r_234041, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_234043 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), getitem___234042, str_234040)
        
        
        # Obtaining the type of the subscript
        int_234044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 45), 'int')
        
        # Obtaining the type of the subscript
        str_234045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'str', 'allvecs')
        # Getting the type of 'r' (line 55)
        r_234046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 32), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___234047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 32), r_234046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_234048 = invoke(stypy.reporting.localization.Localization(__file__, 55, 32), getitem___234047, str_234045)
        
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___234049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 32), subscript_call_result_234048, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_234050 = invoke(stypy.reporting.localization.Localization(__file__, 55, 32), getitem___234049, int_234044)
        
        # Processing the call keyword arguments (line 55)
        kwargs_234051 = {}
        # Getting the type of 'assert_allclose' (line 55)
        assert_allclose_234039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 55)
        assert_allclose_call_result_234052 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert_allclose_234039, *[subscript_call_result_234043, subscript_call_result_234050], **kwargs_234051)
        
        
        # Call to assert_allclose(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Call to sum(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Obtaining the type of the subscript
        int_234055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 41), 'int')
        slice_234056 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 28), int_234055, None, None)
        
        # Obtaining the type of the subscript
        str_234057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'str', 'allvecs')
        # Getting the type of 'r' (line 56)
        r_234058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___234059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 28), r_234058, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_234060 = invoke(stypy.reporting.localization.Localization(__file__, 56, 28), getitem___234059, str_234057)
        
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___234061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 28), subscript_call_result_234060, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_234062 = invoke(stypy.reporting.localization.Localization(__file__, 56, 28), getitem___234061, slice_234056)
        
        # Processing the call keyword arguments (line 56)
        kwargs_234063 = {}
        # Getting the type of 'sum' (line 56)
        sum_234054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'sum', False)
        # Calling sum(args, kwargs) (line 56)
        sum_call_result_234064 = invoke(stypy.reporting.localization.Localization(__file__, 56, 24), sum_234054, *[subscript_call_result_234062], **kwargs_234063)
        
        # Getting the type of 'accumulator' (line 56)
        accumulator_234065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'accumulator', False)
        # Obtaining the member 'accum' of a type (line 56)
        accum_234066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 47), accumulator_234065, 'accum')
        # Processing the call keyword arguments (line 56)
        kwargs_234067 = {}
        # Getting the type of 'assert_allclose' (line 56)
        assert_allclose_234053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 56)
        assert_allclose_call_result_234068 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assert_allclose_234053, *[sum_call_result_234064, accum_234066], **kwargs_234067)
        
        
        # ################# End of 'test_dogleg_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dogleg_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_234069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_234069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dogleg_callback'
        return stypy_return_type_234069


    @norecursion
    def test_solver_concordance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_solver_concordance'
        module_type_store = module_type_store.open_function_context('test_solver_concordance', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_localization', localization)
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_function_name', 'TestTrustRegionSolvers.test_solver_concordance')
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_param_names_list', [])
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTrustRegionSolvers.test_solver_concordance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTrustRegionSolvers.test_solver_concordance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_solver_concordance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_solver_concordance(...)' code ##################

        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'rosen' (line 63)
        rosen_234070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'rosen')
        # Assigning a type to the variable 'f' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'f', rosen_234070)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'rosen_der' (line 64)
        rosen_der_234071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'rosen_der')
        # Assigning a type to the variable 'g' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'g', rosen_der_234071)
        
        # Assigning a Name to a Name (line 65):
        # Getting the type of 'rosen_hess' (line 65)
        rosen_hess_234072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'rosen_hess')
        # Assigning a type to the variable 'h' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'h', rosen_hess_234072)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_234073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        # Getting the type of 'self' (line 66)
        self_234074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'self')
        # Obtaining the member 'easy_guess' of a type (line 66)
        easy_guess_234075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 19), self_234074, 'easy_guess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 19), tuple_234073, easy_guess_234075)
        # Adding element type (line 66)
        # Getting the type of 'self' (line 66)
        self_234076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 36), 'self')
        # Obtaining the member 'hard_guess' of a type (line 66)
        hard_guess_234077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 36), self_234076, 'hard_guess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 19), tuple_234073, hard_guess_234077)
        
        # Testing the type of a for loop iterable (line 66)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 8), tuple_234073)
        # Getting the type of the for loop variable (line 66)
        for_loop_var_234078 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 8), tuple_234073)
        # Assigning a type to the variable 'x0' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'x0', for_loop_var_234078)
        # SSA begins for a for statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 67):
        
        # Call to minimize(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'f' (line 67)
        f_234080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'f', False)
        # Getting the type of 'x0' (line 67)
        x0_234081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 35), 'x0', False)
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'g' (line 67)
        g_234082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'g', False)
        keyword_234083 = g_234082
        # Getting the type of 'h' (line 67)
        h_234084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 51), 'h', False)
        keyword_234085 = h_234084
        float_234086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 58), 'float')
        keyword_234087 = float_234086
        str_234088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'str', 'dogleg')
        keyword_234089 = str_234088
        
        # Obtaining an instance of the builtin type 'dict' (line 68)
        dict_234090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 57), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 68)
        # Adding element type (key, value) (line 68)
        str_234091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 58), 'str', 'return_all')
        # Getting the type of 'True' (line 68)
        True_234092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 72), 'True', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 57), dict_234090, (str_234091, True_234092))
        
        keyword_234093 = dict_234090
        kwargs_234094 = {'hess': keyword_234085, 'options': keyword_234093, 'jac': keyword_234083, 'tol': keyword_234087, 'method': keyword_234089}
        # Getting the type of 'minimize' (line 67)
        minimize_234079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'minimize', False)
        # Calling minimize(args, kwargs) (line 67)
        minimize_call_result_234095 = invoke(stypy.reporting.localization.Localization(__file__, 67, 23), minimize_234079, *[f_234080, x0_234081], **kwargs_234094)
        
        # Assigning a type to the variable 'r_dogleg' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'r_dogleg', minimize_call_result_234095)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to minimize(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'f' (line 69)
        f_234097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 35), 'f', False)
        # Getting the type of 'x0' (line 69)
        x0_234098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'x0', False)
        # Processing the call keyword arguments (line 69)
        # Getting the type of 'g' (line 69)
        g_234099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 46), 'g', False)
        keyword_234100 = g_234099
        # Getting the type of 'h' (line 69)
        h_234101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 54), 'h', False)
        keyword_234102 = h_234101
        float_234103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 61), 'float')
        keyword_234104 = float_234103
        str_234105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 42), 'str', 'trust-ncg')
        keyword_234106 = str_234105
        
        # Obtaining an instance of the builtin type 'dict' (line 71)
        dict_234107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 71)
        # Adding element type (key, value) (line 71)
        str_234108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 44), 'str', 'return_all')
        # Getting the type of 'True' (line 71)
        True_234109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 58), 'True', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 43), dict_234107, (str_234108, True_234109))
        
        keyword_234110 = dict_234107
        kwargs_234111 = {'hess': keyword_234102, 'options': keyword_234110, 'jac': keyword_234100, 'tol': keyword_234104, 'method': keyword_234106}
        # Getting the type of 'minimize' (line 69)
        minimize_234096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'minimize', False)
        # Calling minimize(args, kwargs) (line 69)
        minimize_call_result_234112 = invoke(stypy.reporting.localization.Localization(__file__, 69, 26), minimize_234096, *[f_234097, x0_234098], **kwargs_234111)
        
        # Assigning a type to the variable 'r_trust_ncg' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'r_trust_ncg', minimize_call_result_234112)
        
        # Assigning a Call to a Name (line 72):
        
        # Call to minimize(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'f' (line 72)
        f_234114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 38), 'f', False)
        # Getting the type of 'x0' (line 72)
        x0_234115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 41), 'x0', False)
        # Processing the call keyword arguments (line 72)
        # Getting the type of 'g' (line 72)
        g_234116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 49), 'g', False)
        keyword_234117 = g_234116
        # Getting the type of 'h' (line 72)
        h_234118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 57), 'h', False)
        keyword_234119 = h_234118
        float_234120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 64), 'float')
        keyword_234121 = float_234120
        str_234122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 42), 'str', 'trust-krylov')
        keyword_234123 = str_234122
        
        # Obtaining an instance of the builtin type 'dict' (line 74)
        dict_234124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 74)
        # Adding element type (key, value) (line 74)
        str_234125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 44), 'str', 'return_all')
        # Getting the type of 'True' (line 74)
        True_234126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 58), 'True', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 43), dict_234124, (str_234125, True_234126))
        
        keyword_234127 = dict_234124
        kwargs_234128 = {'hess': keyword_234119, 'options': keyword_234127, 'jac': keyword_234117, 'tol': keyword_234121, 'method': keyword_234123}
        # Getting the type of 'minimize' (line 72)
        minimize_234113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'minimize', False)
        # Calling minimize(args, kwargs) (line 72)
        minimize_call_result_234129 = invoke(stypy.reporting.localization.Localization(__file__, 72, 29), minimize_234113, *[f_234114, x0_234115], **kwargs_234128)
        
        # Assigning a type to the variable 'r_trust_krylov' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'r_trust_krylov', minimize_call_result_234129)
        
        # Assigning a Call to a Name (line 75):
        
        # Call to minimize(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'f' (line 75)
        f_234131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'f', False)
        # Getting the type of 'x0' (line 75)
        x0_234132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'x0', False)
        # Processing the call keyword arguments (line 75)
        # Getting the type of 'g' (line 75)
        g_234133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'g', False)
        keyword_234134 = g_234133
        # Getting the type of 'h' (line 75)
        h_234135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 48), 'h', False)
        keyword_234136 = h_234135
        float_234137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 55), 'float')
        keyword_234138 = float_234137
        str_234139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'str', 'newton-cg')
        keyword_234140 = str_234139
        
        # Obtaining an instance of the builtin type 'dict' (line 76)
        dict_234141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 57), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 76)
        # Adding element type (key, value) (line 76)
        str_234142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 58), 'str', 'return_all')
        # Getting the type of 'True' (line 76)
        True_234143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 72), 'True', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 57), dict_234141, (str_234142, True_234143))
        
        keyword_234144 = dict_234141
        kwargs_234145 = {'hess': keyword_234136, 'options': keyword_234144, 'jac': keyword_234134, 'tol': keyword_234138, 'method': keyword_234140}
        # Getting the type of 'minimize' (line 75)
        minimize_234130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'minimize', False)
        # Calling minimize(args, kwargs) (line 75)
        minimize_call_result_234146 = invoke(stypy.reporting.localization.Localization(__file__, 75, 20), minimize_234130, *[f_234131, x0_234132], **kwargs_234145)
        
        # Assigning a type to the variable 'r_ncg' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'r_ncg', minimize_call_result_234146)
        
        # Assigning a Call to a Name (line 77):
        
        # Call to minimize(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'f' (line 77)
        f_234148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'f', False)
        # Getting the type of 'x0' (line 77)
        x0_234149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'x0', False)
        # Processing the call keyword arguments (line 77)
        # Getting the type of 'g' (line 77)
        g_234150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 46), 'g', False)
        keyword_234151 = g_234150
        # Getting the type of 'h' (line 77)
        h_234152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 54), 'h', False)
        keyword_234153 = h_234152
        float_234154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 61), 'float')
        keyword_234155 = float_234154
        str_234156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'str', 'trust-exact')
        keyword_234157 = str_234156
        
        # Obtaining an instance of the builtin type 'dict' (line 79)
        dict_234158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 79)
        # Adding element type (key, value) (line 79)
        str_234159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 44), 'str', 'return_all')
        # Getting the type of 'True' (line 79)
        True_234160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 58), 'True', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 43), dict_234158, (str_234159, True_234160))
        
        keyword_234161 = dict_234158
        kwargs_234162 = {'hess': keyword_234153, 'options': keyword_234161, 'jac': keyword_234151, 'tol': keyword_234155, 'method': keyword_234157}
        # Getting the type of 'minimize' (line 77)
        minimize_234147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'minimize', False)
        # Calling minimize(args, kwargs) (line 77)
        minimize_call_result_234163 = invoke(stypy.reporting.localization.Localization(__file__, 77, 26), minimize_234147, *[f_234148, x0_234149], **kwargs_234162)
        
        # Assigning a type to the variable 'r_iterative' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'r_iterative', minimize_call_result_234163)
        
        # Call to assert_allclose(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'self' (line 80)
        self_234165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'self', False)
        # Obtaining the member 'x_opt' of a type (line 80)
        x_opt_234166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 28), self_234165, 'x_opt')
        
        # Obtaining the type of the subscript
        str_234167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 49), 'str', 'x')
        # Getting the type of 'r_dogleg' (line 80)
        r_dogleg_234168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 40), 'r_dogleg', False)
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___234169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 40), r_dogleg_234168, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_234170 = invoke(stypy.reporting.localization.Localization(__file__, 80, 40), getitem___234169, str_234167)
        
        # Processing the call keyword arguments (line 80)
        kwargs_234171 = {}
        # Getting the type of 'assert_allclose' (line 80)
        assert_allclose_234164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 80)
        assert_allclose_call_result_234172 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), assert_allclose_234164, *[x_opt_234166, subscript_call_result_234170], **kwargs_234171)
        
        
        # Call to assert_allclose(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'self' (line 81)
        self_234174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'self', False)
        # Obtaining the member 'x_opt' of a type (line 81)
        x_opt_234175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 28), self_234174, 'x_opt')
        
        # Obtaining the type of the subscript
        str_234176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 52), 'str', 'x')
        # Getting the type of 'r_trust_ncg' (line 81)
        r_trust_ncg_234177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'r_trust_ncg', False)
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___234178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 40), r_trust_ncg_234177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_234179 = invoke(stypy.reporting.localization.Localization(__file__, 81, 40), getitem___234178, str_234176)
        
        # Processing the call keyword arguments (line 81)
        kwargs_234180 = {}
        # Getting the type of 'assert_allclose' (line 81)
        assert_allclose_234173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 81)
        assert_allclose_call_result_234181 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), assert_allclose_234173, *[x_opt_234175, subscript_call_result_234179], **kwargs_234180)
        
        
        # Call to assert_allclose(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'self' (line 82)
        self_234183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'self', False)
        # Obtaining the member 'x_opt' of a type (line 82)
        x_opt_234184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 28), self_234183, 'x_opt')
        
        # Obtaining the type of the subscript
        str_234185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 55), 'str', 'x')
        # Getting the type of 'r_trust_krylov' (line 82)
        r_trust_krylov_234186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'r_trust_krylov', False)
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___234187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), r_trust_krylov_234186, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_234188 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___234187, str_234185)
        
        # Processing the call keyword arguments (line 82)
        kwargs_234189 = {}
        # Getting the type of 'assert_allclose' (line 82)
        assert_allclose_234182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 82)
        assert_allclose_call_result_234190 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), assert_allclose_234182, *[x_opt_234184, subscript_call_result_234188], **kwargs_234189)
        
        
        # Call to assert_allclose(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'self' (line 83)
        self_234192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'self', False)
        # Obtaining the member 'x_opt' of a type (line 83)
        x_opt_234193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 28), self_234192, 'x_opt')
        
        # Obtaining the type of the subscript
        str_234194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 46), 'str', 'x')
        # Getting the type of 'r_ncg' (line 83)
        r_ncg_234195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'r_ncg', False)
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___234196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 40), r_ncg_234195, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_234197 = invoke(stypy.reporting.localization.Localization(__file__, 83, 40), getitem___234196, str_234194)
        
        # Processing the call keyword arguments (line 83)
        kwargs_234198 = {}
        # Getting the type of 'assert_allclose' (line 83)
        assert_allclose_234191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 83)
        assert_allclose_call_result_234199 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), assert_allclose_234191, *[x_opt_234193, subscript_call_result_234197], **kwargs_234198)
        
        
        # Call to assert_allclose(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'self' (line 84)
        self_234201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'self', False)
        # Obtaining the member 'x_opt' of a type (line 84)
        x_opt_234202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 28), self_234201, 'x_opt')
        
        # Obtaining the type of the subscript
        str_234203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 52), 'str', 'x')
        # Getting the type of 'r_iterative' (line 84)
        r_iterative_234204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 40), 'r_iterative', False)
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___234205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 40), r_iterative_234204, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_234206 = invoke(stypy.reporting.localization.Localization(__file__, 84, 40), getitem___234205, str_234203)
        
        # Processing the call keyword arguments (line 84)
        kwargs_234207 = {}
        # Getting the type of 'assert_allclose' (line 84)
        assert_allclose_234200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 84)
        assert_allclose_call_result_234208 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), assert_allclose_234200, *[x_opt_234202, subscript_call_result_234206], **kwargs_234207)
        
        
        # Call to assert_(...): (line 85)
        # Processing the call arguments (line 85)
        
        
        # Call to len(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining the type of the subscript
        str_234211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 33), 'str', 'allvecs')
        # Getting the type of 'r_dogleg' (line 85)
        r_dogleg_234212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'r_dogleg', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___234213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), r_dogleg_234212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_234214 = invoke(stypy.reporting.localization.Localization(__file__, 85, 24), getitem___234213, str_234211)
        
        # Processing the call keyword arguments (line 85)
        kwargs_234215 = {}
        # Getting the type of 'len' (line 85)
        len_234210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'len', False)
        # Calling len(args, kwargs) (line 85)
        len_call_result_234216 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), len_234210, *[subscript_call_result_234214], **kwargs_234215)
        
        
        # Call to len(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining the type of the subscript
        str_234218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 57), 'str', 'allvecs')
        # Getting the type of 'r_ncg' (line 85)
        r_ncg_234219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 51), 'r_ncg', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___234220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 51), r_ncg_234219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_234221 = invoke(stypy.reporting.localization.Localization(__file__, 85, 51), getitem___234220, str_234218)
        
        # Processing the call keyword arguments (line 85)
        kwargs_234222 = {}
        # Getting the type of 'len' (line 85)
        len_234217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 47), 'len', False)
        # Calling len(args, kwargs) (line 85)
        len_call_result_234223 = invoke(stypy.reporting.localization.Localization(__file__, 85, 47), len_234217, *[subscript_call_result_234221], **kwargs_234222)
        
        # Applying the binary operator '<' (line 85)
        result_lt_234224 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 20), '<', len_call_result_234216, len_call_result_234223)
        
        # Processing the call keyword arguments (line 85)
        kwargs_234225 = {}
        # Getting the type of 'assert_' (line 85)
        assert__234209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 85)
        assert__call_result_234226 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), assert__234209, *[result_lt_234224], **kwargs_234225)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_solver_concordance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_solver_concordance' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_234227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_234227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_solver_concordance'
        return stypy_return_type_234227


    @norecursion
    def test_trust_ncg_hessp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_trust_ncg_hessp'
        module_type_store = module_type_store.open_function_context('test_trust_ncg_hessp', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_localization', localization)
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_function_name', 'TestTrustRegionSolvers.test_trust_ncg_hessp')
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_param_names_list', [])
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTrustRegionSolvers.test_trust_ncg_hessp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTrustRegionSolvers.test_trust_ncg_hessp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_trust_ncg_hessp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_trust_ncg_hessp(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_234228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        # Getting the type of 'self' (line 88)
        self_234229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'self')
        # Obtaining the member 'easy_guess' of a type (line 88)
        easy_guess_234230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 19), self_234229, 'easy_guess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), tuple_234228, easy_guess_234230)
        # Adding element type (line 88)
        # Getting the type of 'self' (line 88)
        self_234231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'self')
        # Obtaining the member 'hard_guess' of a type (line 88)
        hard_guess_234232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 36), self_234231, 'hard_guess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), tuple_234228, hard_guess_234232)
        
        # Testing the type of a for loop iterable (line 88)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), tuple_234228)
        # Getting the type of the for loop variable (line 88)
        for_loop_var_234233 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), tuple_234228)
        # Assigning a type to the variable 'x0' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'x0', for_loop_var_234233)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 89):
        
        # Call to minimize(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'rosen' (line 89)
        rosen_234235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'rosen', False)
        # Getting the type of 'x0' (line 89)
        x0_234236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 32), 'x0', False)
        # Processing the call keyword arguments (line 89)
        # Getting the type of 'rosen_der' (line 89)
        rosen_der_234237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'rosen_der', False)
        keyword_234238 = rosen_der_234237
        # Getting the type of 'rosen_hess_prod' (line 89)
        rosen_hess_prod_234239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 57), 'rosen_hess_prod', False)
        keyword_234240 = rosen_hess_prod_234239
        float_234241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'float')
        keyword_234242 = float_234241
        str_234243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 42), 'str', 'trust-ncg')
        keyword_234244 = str_234243
        kwargs_234245 = {'hessp': keyword_234240, 'jac': keyword_234238, 'tol': keyword_234242, 'method': keyword_234244}
        # Getting the type of 'minimize' (line 89)
        minimize_234234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'minimize', False)
        # Calling minimize(args, kwargs) (line 89)
        minimize_call_result_234246 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), minimize_234234, *[rosen_234235, x0_234236], **kwargs_234245)
        
        # Assigning a type to the variable 'r' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'r', minimize_call_result_234246)
        
        # Call to assert_allclose(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'self' (line 91)
        self_234248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'self', False)
        # Obtaining the member 'x_opt' of a type (line 91)
        x_opt_234249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 28), self_234248, 'x_opt')
        
        # Obtaining the type of the subscript
        str_234250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 42), 'str', 'x')
        # Getting the type of 'r' (line 91)
        r_234251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 40), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___234252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 40), r_234251, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_234253 = invoke(stypy.reporting.localization.Localization(__file__, 91, 40), getitem___234252, str_234250)
        
        # Processing the call keyword arguments (line 91)
        kwargs_234254 = {}
        # Getting the type of 'assert_allclose' (line 91)
        assert_allclose_234247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 91)
        assert_allclose_call_result_234255 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), assert_allclose_234247, *[x_opt_234249, subscript_call_result_234253], **kwargs_234254)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_trust_ncg_hessp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_trust_ncg_hessp' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_234256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_234256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_trust_ncg_hessp'
        return stypy_return_type_234256


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 0, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTrustRegionSolvers.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTrustRegionSolvers' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'TestTrustRegionSolvers', TestTrustRegionSolvers)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
