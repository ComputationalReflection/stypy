
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import math
4: import numpy as np
5: 
6: from numpy.testing import assert_allclose, assert_
7: 
8: from scipy.optimize import fmin_cobyla, minimize
9: 
10: 
11: class TestCobyla(object):
12:     def setup_method(self):
13:         self.x0 = [4.95, 0.66]
14:         self.solution = [math.sqrt(25 - (2.0/3)**2), 2.0/3]
15:         self.opts = {'disp': False, 'rhobeg': 1, 'tol': 1e-5,
16:                      'maxiter': 100}
17: 
18:     def fun(self, x):
19:         return x[0]**2 + abs(x[1])**3
20: 
21:     def con1(self, x):
22:         return x[0]**2 + x[1]**2 - 25
23: 
24:     def con2(self, x):
25:         return -self.con1(x)
26: 
27:     def test_simple(self):
28:         x = fmin_cobyla(self.fun, self.x0, [self.con1, self.con2], rhobeg=1,
29:                         rhoend=1e-5, maxfun=100)
30:         assert_allclose(x, self.solution, atol=1e-4)
31: 
32:     def test_minimize_simple(self):
33:         # Minimize with method='COBYLA'
34:         cons = ({'type': 'ineq', 'fun': self.con1},
35:                 {'type': 'ineq', 'fun': self.con2})
36:         sol = minimize(self.fun, self.x0, method='cobyla', constraints=cons,
37:                        options=self.opts)
38:         assert_allclose(sol.x, self.solution, atol=1e-4)
39:         assert_(sol.success, sol.message)
40:         assert_(sol.maxcv < 1e-5, sol)
41:         assert_(sol.nfev < 70, sol)
42:         assert_(sol.fun < self.fun(self.solution) + 1e-3, sol)
43: 
44:     def test_minimize_constraint_violation(self):
45:         np.random.seed(1234)
46:         pb = np.random.rand(10, 10)
47:         spread = np.random.rand(10)
48: 
49:         def p(w):
50:             return pb.dot(w)
51: 
52:         def f(w):
53:             return -(w * spread).sum()
54: 
55:         def c1(w):
56:             return 500 - abs(p(w)).sum()
57: 
58:         def c2(w):
59:             return 5 - abs(p(w).sum())
60: 
61:         def c3(w):
62:             return 5 - abs(p(w)).max()
63: 
64:         cons = ({'type': 'ineq', 'fun': c1},
65:                 {'type': 'ineq', 'fun': c2},
66:                 {'type': 'ineq', 'fun': c3})
67:         w0 = np.zeros((10, 1))
68:         sol = minimize(f, w0, method='cobyla', constraints=cons,
69:                        options={'catol': 1e-6})
70:         assert_(sol.maxcv > 1e-6)
71:         assert_(not sol.success)
72: 
73: 
74: def test_vector_constraints():
75:     # test that fmin_cobyla and minimize can take a combination
76:     # of constraints, some returning a number and others an array
77:     def fun(x):
78:         return (x[0] - 1)**2 + (x[1] - 2.5)**2
79: 
80:     def fmin(x):
81:         return fun(x) - 1
82: 
83:     def cons1(x):
84:         a = np.array([[1, -2, 2], [-1, -2, 6], [-1, 2, 2]])
85:         return np.array([a[i, 0] * x[0] + a[i, 1] * x[1] +
86:                          a[i, 2] for i in range(len(a))])
87: 
88:     def cons2(x):
89:         return x     # identity, acts as bounds x > 0
90: 
91:     x0 = np.array([2, 0])
92:     cons_list = [fun, cons1, cons2]
93: 
94:     xsol = [1.4, 1.7]
95:     fsol = 0.8
96: 
97:     # testing fmin_cobyla
98:     sol = fmin_cobyla(fun, x0, cons_list, rhoend=1e-5)
99:     assert_allclose(sol, xsol, atol=1e-4)
100: 
101:     sol = fmin_cobyla(fun, x0, fmin, rhoend=1e-5)
102:     assert_allclose(fun(sol), 1, atol=1e-4)
103: 
104:     # testing minimize
105:     constraints = [{'type': 'ineq', 'fun': cons} for cons in cons_list]
106:     sol = minimize(fun, x0, constraints=constraints, tol=1e-5)
107:     assert_allclose(sol.x, xsol, atol=1e-4)
108:     assert_(sol.success, sol.message)
109:     assert_allclose(sol.fun, fsol, atol=1e-4)
110: 
111:     constraints = {'type': 'ineq', 'fun': fmin}
112:     sol = minimize(fun, x0, constraints=constraints, tol=1e-5)
113:     assert_allclose(sol.fun, 1, atol=1e-4)
114: 
115: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import math' statement (line 3)
import math

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_204787 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_204787) is not StypyTypeError):

    if (import_204787 != 'pyd_module'):
        __import__(import_204787)
        sys_modules_204788 = sys.modules[import_204787]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_204788.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_204787)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_allclose, assert_' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_204789 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_204789) is not StypyTypeError):

    if (import_204789 != 'pyd_module'):
        __import__(import_204789)
        sys_modules_204790 = sys.modules[import_204789]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_204790.module_type_store, module_type_store, ['assert_allclose', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_204790, sys_modules_204790.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_'], [assert_allclose, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_204789)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize import fmin_cobyla, minimize' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_204791 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize')

if (type(import_204791) is not StypyTypeError):

    if (import_204791 != 'pyd_module'):
        __import__(import_204791)
        sys_modules_204792 = sys.modules[import_204791]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', sys_modules_204792.module_type_store, module_type_store, ['fmin_cobyla', 'minimize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_204792, sys_modules_204792.module_type_store, module_type_store)
    else:
        from scipy.optimize import fmin_cobyla, minimize

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', None, module_type_store, ['fmin_cobyla', 'minimize'], [fmin_cobyla, minimize])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', import_204791)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'TestCobyla' class

class TestCobyla(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCobyla.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestCobyla.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCobyla.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCobyla.setup_method.__dict__.__setitem__('stypy_function_name', 'TestCobyla.setup_method')
        TestCobyla.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestCobyla.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCobyla.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCobyla.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCobyla.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCobyla.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCobyla.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCobyla.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 13):
        
        # Obtaining an instance of the builtin type 'list' (line 13)
        list_204793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 13)
        # Adding element type (line 13)
        float_204794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), list_204793, float_204794)
        # Adding element type (line 13)
        float_204795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), list_204793, float_204795)
        
        # Getting the type of 'self' (line 13)
        self_204796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self')
        # Setting the type of the member 'x0' of a type (line 13)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_204796, 'x0', list_204793)
        
        # Assigning a List to a Attribute (line 14):
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_204797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        
        # Call to sqrt(...): (line 14)
        # Processing the call arguments (line 14)
        int_204800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'int')
        float_204801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 41), 'float')
        int_204802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 45), 'int')
        # Applying the binary operator 'div' (line 14)
        result_div_204803 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 41), 'div', float_204801, int_204802)
        
        int_204804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 49), 'int')
        # Applying the binary operator '**' (line 14)
        result_pow_204805 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 40), '**', result_div_204803, int_204804)
        
        # Applying the binary operator '-' (line 14)
        result_sub_204806 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 35), '-', int_204800, result_pow_204805)
        
        # Processing the call keyword arguments (line 14)
        kwargs_204807 = {}
        # Getting the type of 'math' (line 14)
        math_204798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 25), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 14)
        sqrt_204799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 25), math_204798, 'sqrt')
        # Calling sqrt(args, kwargs) (line 14)
        sqrt_call_result_204808 = invoke(stypy.reporting.localization.Localization(__file__, 14, 25), sqrt_204799, *[result_sub_204806], **kwargs_204807)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), list_204797, sqrt_call_result_204808)
        # Adding element type (line 14)
        float_204809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 53), 'float')
        int_204810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 57), 'int')
        # Applying the binary operator 'div' (line 14)
        result_div_204811 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 53), 'div', float_204809, int_204810)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), list_204797, result_div_204811)
        
        # Getting the type of 'self' (line 14)
        self_204812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'solution' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_204812, 'solution', list_204797)
        
        # Assigning a Dict to a Attribute (line 15):
        
        # Obtaining an instance of the builtin type 'dict' (line 15)
        dict_204813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 15)
        # Adding element type (key, value) (line 15)
        str_204814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'str', 'disp')
        # Getting the type of 'False' (line 15)
        False_204815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 29), 'False')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), dict_204813, (str_204814, False_204815))
        # Adding element type (key, value) (line 15)
        str_204816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 36), 'str', 'rhobeg')
        int_204817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 46), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), dict_204813, (str_204816, int_204817))
        # Adding element type (key, value) (line 15)
        str_204818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 49), 'str', 'tol')
        float_204819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 56), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), dict_204813, (str_204818, float_204819))
        # Adding element type (key, value) (line 15)
        str_204820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'str', 'maxiter')
        int_204821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), dict_204813, (str_204820, int_204821))
        
        # Getting the type of 'self' (line 15)
        self_204822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'opts' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_204822, 'opts', dict_204813)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_204823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_204823


    @norecursion
    def fun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun'
        module_type_store = module_type_store.open_function_context('fun', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCobyla.fun.__dict__.__setitem__('stypy_localization', localization)
        TestCobyla.fun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCobyla.fun.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCobyla.fun.__dict__.__setitem__('stypy_function_name', 'TestCobyla.fun')
        TestCobyla.fun.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestCobyla.fun.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCobyla.fun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCobyla.fun.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCobyla.fun.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCobyla.fun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCobyla.fun.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCobyla.fun', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun(...)' code ##################

        
        # Obtaining the type of the subscript
        int_204824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
        # Getting the type of 'x' (line 19)
        x_204825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'x')
        # Obtaining the member '__getitem__' of a type (line 19)
        getitem___204826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), x_204825, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 19)
        subscript_call_result_204827 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), getitem___204826, int_204824)
        
        int_204828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'int')
        # Applying the binary operator '**' (line 19)
        result_pow_204829 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 15), '**', subscript_call_result_204827, int_204828)
        
        
        # Call to abs(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Obtaining the type of the subscript
        int_204831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
        # Getting the type of 'x' (line 19)
        x_204832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 19)
        getitem___204833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 29), x_204832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 19)
        subscript_call_result_204834 = invoke(stypy.reporting.localization.Localization(__file__, 19, 29), getitem___204833, int_204831)
        
        # Processing the call keyword arguments (line 19)
        kwargs_204835 = {}
        # Getting the type of 'abs' (line 19)
        abs_204830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'abs', False)
        # Calling abs(args, kwargs) (line 19)
        abs_call_result_204836 = invoke(stypy.reporting.localization.Localization(__file__, 19, 25), abs_204830, *[subscript_call_result_204834], **kwargs_204835)
        
        int_204837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'int')
        # Applying the binary operator '**' (line 19)
        result_pow_204838 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 25), '**', abs_call_result_204836, int_204837)
        
        # Applying the binary operator '+' (line 19)
        result_add_204839 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 15), '+', result_pow_204829, result_pow_204838)
        
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', result_add_204839)
        
        # ################# End of 'fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_204840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun'
        return stypy_return_type_204840


    @norecursion
    def con1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'con1'
        module_type_store = module_type_store.open_function_context('con1', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCobyla.con1.__dict__.__setitem__('stypy_localization', localization)
        TestCobyla.con1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCobyla.con1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCobyla.con1.__dict__.__setitem__('stypy_function_name', 'TestCobyla.con1')
        TestCobyla.con1.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestCobyla.con1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCobyla.con1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCobyla.con1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCobyla.con1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCobyla.con1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCobyla.con1.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCobyla.con1', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'con1', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'con1(...)' code ##################

        
        # Obtaining the type of the subscript
        int_204841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
        # Getting the type of 'x' (line 22)
        x_204842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'x')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___204843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 15), x_204842, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_204844 = invoke(stypy.reporting.localization.Localization(__file__, 22, 15), getitem___204843, int_204841)
        
        int_204845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
        # Applying the binary operator '**' (line 22)
        result_pow_204846 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '**', subscript_call_result_204844, int_204845)
        
        
        # Obtaining the type of the subscript
        int_204847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'int')
        # Getting the type of 'x' (line 22)
        x_204848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'x')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___204849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), x_204848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_204850 = invoke(stypy.reporting.localization.Localization(__file__, 22, 25), getitem___204849, int_204847)
        
        int_204851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'int')
        # Applying the binary operator '**' (line 22)
        result_pow_204852 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 25), '**', subscript_call_result_204850, int_204851)
        
        # Applying the binary operator '+' (line 22)
        result_add_204853 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '+', result_pow_204846, result_pow_204852)
        
        int_204854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'int')
        # Applying the binary operator '-' (line 22)
        result_sub_204855 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 33), '-', result_add_204853, int_204854)
        
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', result_sub_204855)
        
        # ################# End of 'con1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'con1' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_204856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204856)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'con1'
        return stypy_return_type_204856


    @norecursion
    def con2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'con2'
        module_type_store = module_type_store.open_function_context('con2', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCobyla.con2.__dict__.__setitem__('stypy_localization', localization)
        TestCobyla.con2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCobyla.con2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCobyla.con2.__dict__.__setitem__('stypy_function_name', 'TestCobyla.con2')
        TestCobyla.con2.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestCobyla.con2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCobyla.con2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCobyla.con2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCobyla.con2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCobyla.con2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCobyla.con2.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCobyla.con2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'con2', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'con2(...)' code ##################

        
        
        # Call to con1(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'x' (line 25)
        x_204859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'x', False)
        # Processing the call keyword arguments (line 25)
        kwargs_204860 = {}
        # Getting the type of 'self' (line 25)
        self_204857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self', False)
        # Obtaining the member 'con1' of a type (line 25)
        con1_204858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_204857, 'con1')
        # Calling con1(args, kwargs) (line 25)
        con1_call_result_204861 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), con1_204858, *[x_204859], **kwargs_204860)
        
        # Applying the 'usub' unary operator (line 25)
        result___neg___204862 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), 'usub', con1_call_result_204861)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', result___neg___204862)
        
        # ################# End of 'con2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'con2' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_204863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204863)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'con2'
        return stypy_return_type_204863


    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCobyla.test_simple.__dict__.__setitem__('stypy_localization', localization)
        TestCobyla.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCobyla.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCobyla.test_simple.__dict__.__setitem__('stypy_function_name', 'TestCobyla.test_simple')
        TestCobyla.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        TestCobyla.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCobyla.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCobyla.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCobyla.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCobyla.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCobyla.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCobyla.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 28):
        
        # Call to fmin_cobyla(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_204865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'self', False)
        # Obtaining the member 'fun' of a type (line 28)
        fun_204866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), self_204865, 'fun')
        # Getting the type of 'self' (line 28)
        self_204867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'self', False)
        # Obtaining the member 'x0' of a type (line 28)
        x0_204868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 34), self_204867, 'x0')
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_204869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'self' (line 28)
        self_204870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 44), 'self', False)
        # Obtaining the member 'con1' of a type (line 28)
        con1_204871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 44), self_204870, 'con1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 43), list_204869, con1_204871)
        # Adding element type (line 28)
        # Getting the type of 'self' (line 28)
        self_204872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 55), 'self', False)
        # Obtaining the member 'con2' of a type (line 28)
        con2_204873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 55), self_204872, 'con2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 43), list_204869, con2_204873)
        
        # Processing the call keyword arguments (line 28)
        int_204874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 74), 'int')
        keyword_204875 = int_204874
        float_204876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 31), 'float')
        keyword_204877 = float_204876
        int_204878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 44), 'int')
        keyword_204879 = int_204878
        kwargs_204880 = {'rhoend': keyword_204877, 'maxfun': keyword_204879, 'rhobeg': keyword_204875}
        # Getting the type of 'fmin_cobyla' (line 28)
        fmin_cobyla_204864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'fmin_cobyla', False)
        # Calling fmin_cobyla(args, kwargs) (line 28)
        fmin_cobyla_call_result_204881 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), fmin_cobyla_204864, *[fun_204866, x0_204868, list_204869], **kwargs_204880)
        
        # Assigning a type to the variable 'x' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'x', fmin_cobyla_call_result_204881)
        
        # Call to assert_allclose(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'x' (line 30)
        x_204883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'x', False)
        # Getting the type of 'self' (line 30)
        self_204884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'self', False)
        # Obtaining the member 'solution' of a type (line 30)
        solution_204885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 27), self_204884, 'solution')
        # Processing the call keyword arguments (line 30)
        float_204886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 47), 'float')
        keyword_204887 = float_204886
        kwargs_204888 = {'atol': keyword_204887}
        # Getting the type of 'assert_allclose' (line 30)
        assert_allclose_204882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 30)
        assert_allclose_call_result_204889 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_allclose_204882, *[x_204883, solution_204885], **kwargs_204888)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_204890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204890)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_204890


    @norecursion
    def test_minimize_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_simple'
        module_type_store = module_type_store.open_function_context('test_minimize_simple', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_localization', localization)
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_function_name', 'TestCobyla.test_minimize_simple')
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_param_names_list', [])
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCobyla.test_minimize_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCobyla.test_minimize_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_simple(...)' code ##################

        
        # Assigning a Tuple to a Name (line 34):
        
        # Obtaining an instance of the builtin type 'tuple' (line 34)
        tuple_204891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 34)
        # Adding element type (line 34)
        
        # Obtaining an instance of the builtin type 'dict' (line 34)
        dict_204892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 34)
        # Adding element type (key, value) (line 34)
        str_204893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'str', 'type')
        str_204894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), dict_204892, (str_204893, str_204894))
        # Adding element type (key, value) (line 34)
        str_204895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 33), 'str', 'fun')
        # Getting the type of 'self' (line 34)
        self_204896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 40), 'self')
        # Obtaining the member 'con1' of a type (line 34)
        con1_204897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 40), self_204896, 'con1')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), dict_204892, (str_204895, con1_204897))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), tuple_204891, dict_204892)
        # Adding element type (line 34)
        
        # Obtaining an instance of the builtin type 'dict' (line 35)
        dict_204898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 35)
        # Adding element type (key, value) (line 35)
        str_204899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'str', 'type')
        str_204900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 16), dict_204898, (str_204899, str_204900))
        # Adding element type (key, value) (line 35)
        str_204901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'str', 'fun')
        # Getting the type of 'self' (line 35)
        self_204902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'self')
        # Obtaining the member 'con2' of a type (line 35)
        con2_204903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 40), self_204902, 'con2')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 16), dict_204898, (str_204901, con2_204903))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), tuple_204891, dict_204898)
        
        # Assigning a type to the variable 'cons' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'cons', tuple_204891)
        
        # Assigning a Call to a Name (line 36):
        
        # Call to minimize(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_204905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 36)
        fun_204906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), self_204905, 'fun')
        # Getting the type of 'self' (line 36)
        self_204907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'self', False)
        # Obtaining the member 'x0' of a type (line 36)
        x0_204908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 33), self_204907, 'x0')
        # Processing the call keyword arguments (line 36)
        str_204909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 49), 'str', 'cobyla')
        keyword_204910 = str_204909
        # Getting the type of 'cons' (line 36)
        cons_204911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 71), 'cons', False)
        keyword_204912 = cons_204911
        # Getting the type of 'self' (line 37)
        self_204913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'self', False)
        # Obtaining the member 'opts' of a type (line 37)
        opts_204914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 31), self_204913, 'opts')
        keyword_204915 = opts_204914
        kwargs_204916 = {'options': keyword_204915, 'method': keyword_204910, 'constraints': keyword_204912}
        # Getting the type of 'minimize' (line 36)
        minimize_204904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 36)
        minimize_call_result_204917 = invoke(stypy.reporting.localization.Localization(__file__, 36, 14), minimize_204904, *[fun_204906, x0_204908], **kwargs_204916)
        
        # Assigning a type to the variable 'sol' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'sol', minimize_call_result_204917)
        
        # Call to assert_allclose(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'sol' (line 38)
        sol_204919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 38)
        x_204920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), sol_204919, 'x')
        # Getting the type of 'self' (line 38)
        self_204921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'self', False)
        # Obtaining the member 'solution' of a type (line 38)
        solution_204922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 31), self_204921, 'solution')
        # Processing the call keyword arguments (line 38)
        float_204923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 51), 'float')
        keyword_204924 = float_204923
        kwargs_204925 = {'atol': keyword_204924}
        # Getting the type of 'assert_allclose' (line 38)
        assert_allclose_204918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 38)
        assert_allclose_call_result_204926 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assert_allclose_204918, *[x_204920, solution_204922], **kwargs_204925)
        
        
        # Call to assert_(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'sol' (line 39)
        sol_204928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 39)
        success_204929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), sol_204928, 'success')
        # Getting the type of 'sol' (line 39)
        sol_204930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'sol', False)
        # Obtaining the member 'message' of a type (line 39)
        message_204931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), sol_204930, 'message')
        # Processing the call keyword arguments (line 39)
        kwargs_204932 = {}
        # Getting the type of 'assert_' (line 39)
        assert__204927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 39)
        assert__call_result_204933 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert__204927, *[success_204929, message_204931], **kwargs_204932)
        
        
        # Call to assert_(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Getting the type of 'sol' (line 40)
        sol_204935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'sol', False)
        # Obtaining the member 'maxcv' of a type (line 40)
        maxcv_204936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), sol_204935, 'maxcv')
        float_204937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'float')
        # Applying the binary operator '<' (line 40)
        result_lt_204938 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), '<', maxcv_204936, float_204937)
        
        # Getting the type of 'sol' (line 40)
        sol_204939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'sol', False)
        # Processing the call keyword arguments (line 40)
        kwargs_204940 = {}
        # Getting the type of 'assert_' (line 40)
        assert__204934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 40)
        assert__call_result_204941 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assert__204934, *[result_lt_204938, sol_204939], **kwargs_204940)
        
        
        # Call to assert_(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Getting the type of 'sol' (line 41)
        sol_204943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'sol', False)
        # Obtaining the member 'nfev' of a type (line 41)
        nfev_204944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), sol_204943, 'nfev')
        int_204945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'int')
        # Applying the binary operator '<' (line 41)
        result_lt_204946 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 16), '<', nfev_204944, int_204945)
        
        # Getting the type of 'sol' (line 41)
        sol_204947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'sol', False)
        # Processing the call keyword arguments (line 41)
        kwargs_204948 = {}
        # Getting the type of 'assert_' (line 41)
        assert__204942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 41)
        assert__call_result_204949 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assert__204942, *[result_lt_204946, sol_204947], **kwargs_204948)
        
        
        # Call to assert_(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Getting the type of 'sol' (line 42)
        sol_204951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'sol', False)
        # Obtaining the member 'fun' of a type (line 42)
        fun_204952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), sol_204951, 'fun')
        
        # Call to fun(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_204955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 35), 'self', False)
        # Obtaining the member 'solution' of a type (line 42)
        solution_204956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 35), self_204955, 'solution')
        # Processing the call keyword arguments (line 42)
        kwargs_204957 = {}
        # Getting the type of 'self' (line 42)
        self_204953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'self', False)
        # Obtaining the member 'fun' of a type (line 42)
        fun_204954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 26), self_204953, 'fun')
        # Calling fun(args, kwargs) (line 42)
        fun_call_result_204958 = invoke(stypy.reporting.localization.Localization(__file__, 42, 26), fun_204954, *[solution_204956], **kwargs_204957)
        
        float_204959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 52), 'float')
        # Applying the binary operator '+' (line 42)
        result_add_204960 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 26), '+', fun_call_result_204958, float_204959)
        
        # Applying the binary operator '<' (line 42)
        result_lt_204961 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 16), '<', fun_204952, result_add_204960)
        
        # Getting the type of 'sol' (line 42)
        sol_204962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 58), 'sol', False)
        # Processing the call keyword arguments (line 42)
        kwargs_204963 = {}
        # Getting the type of 'assert_' (line 42)
        assert__204950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 42)
        assert__call_result_204964 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert__204950, *[result_lt_204961, sol_204962], **kwargs_204963)
        
        
        # ################# End of 'test_minimize_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_204965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_simple'
        return stypy_return_type_204965


    @norecursion
    def test_minimize_constraint_violation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_constraint_violation'
        module_type_store = module_type_store.open_function_context('test_minimize_constraint_violation', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_localization', localization)
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_function_name', 'TestCobyla.test_minimize_constraint_violation')
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_param_names_list', [])
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCobyla.test_minimize_constraint_violation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCobyla.test_minimize_constraint_violation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_constraint_violation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_constraint_violation(...)' code ##################

        
        # Call to seed(...): (line 45)
        # Processing the call arguments (line 45)
        int_204969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_204970 = {}
        # Getting the type of 'np' (line 45)
        np_204966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 45)
        random_204967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), np_204966, 'random')
        # Obtaining the member 'seed' of a type (line 45)
        seed_204968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), random_204967, 'seed')
        # Calling seed(args, kwargs) (line 45)
        seed_call_result_204971 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), seed_204968, *[int_204969], **kwargs_204970)
        
        
        # Assigning a Call to a Name (line 46):
        
        # Call to rand(...): (line 46)
        # Processing the call arguments (line 46)
        int_204975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'int')
        int_204976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 32), 'int')
        # Processing the call keyword arguments (line 46)
        kwargs_204977 = {}
        # Getting the type of 'np' (line 46)
        np_204972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'np', False)
        # Obtaining the member 'random' of a type (line 46)
        random_204973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), np_204972, 'random')
        # Obtaining the member 'rand' of a type (line 46)
        rand_204974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), random_204973, 'rand')
        # Calling rand(args, kwargs) (line 46)
        rand_call_result_204978 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), rand_204974, *[int_204975, int_204976], **kwargs_204977)
        
        # Assigning a type to the variable 'pb' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'pb', rand_call_result_204978)
        
        # Assigning a Call to a Name (line 47):
        
        # Call to rand(...): (line 47)
        # Processing the call arguments (line 47)
        int_204982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 32), 'int')
        # Processing the call keyword arguments (line 47)
        kwargs_204983 = {}
        # Getting the type of 'np' (line 47)
        np_204979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'np', False)
        # Obtaining the member 'random' of a type (line 47)
        random_204980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), np_204979, 'random')
        # Obtaining the member 'rand' of a type (line 47)
        rand_204981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), random_204980, 'rand')
        # Calling rand(args, kwargs) (line 47)
        rand_call_result_204984 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), rand_204981, *[int_204982], **kwargs_204983)
        
        # Assigning a type to the variable 'spread' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'spread', rand_call_result_204984)

        @norecursion
        def p(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'p'
            module_type_store = module_type_store.open_function_context('p', 49, 8, False)
            
            # Passed parameters checking function
            p.stypy_localization = localization
            p.stypy_type_of_self = None
            p.stypy_type_store = module_type_store
            p.stypy_function_name = 'p'
            p.stypy_param_names_list = ['w']
            p.stypy_varargs_param_name = None
            p.stypy_kwargs_param_name = None
            p.stypy_call_defaults = defaults
            p.stypy_call_varargs = varargs
            p.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'p', ['w'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'p', localization, ['w'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'p(...)' code ##################

            
            # Call to dot(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'w' (line 50)
            w_204987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'w', False)
            # Processing the call keyword arguments (line 50)
            kwargs_204988 = {}
            # Getting the type of 'pb' (line 50)
            pb_204985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'pb', False)
            # Obtaining the member 'dot' of a type (line 50)
            dot_204986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), pb_204985, 'dot')
            # Calling dot(args, kwargs) (line 50)
            dot_call_result_204989 = invoke(stypy.reporting.localization.Localization(__file__, 50, 19), dot_204986, *[w_204987], **kwargs_204988)
            
            # Assigning a type to the variable 'stypy_return_type' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'stypy_return_type', dot_call_result_204989)
            
            # ################# End of 'p(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'p' in the type store
            # Getting the type of 'stypy_return_type' (line 49)
            stypy_return_type_204990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_204990)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'p'
            return stypy_return_type_204990

        # Assigning a type to the variable 'p' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'p', p)

        @norecursion
        def f(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'f'
            module_type_store = module_type_store.open_function_context('f', 52, 8, False)
            
            # Passed parameters checking function
            f.stypy_localization = localization
            f.stypy_type_of_self = None
            f.stypy_type_store = module_type_store
            f.stypy_function_name = 'f'
            f.stypy_param_names_list = ['w']
            f.stypy_varargs_param_name = None
            f.stypy_kwargs_param_name = None
            f.stypy_call_defaults = defaults
            f.stypy_call_varargs = varargs
            f.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'f', ['w'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'f', localization, ['w'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'f(...)' code ##################

            
            
            # Call to sum(...): (line 53)
            # Processing the call keyword arguments (line 53)
            kwargs_204995 = {}
            # Getting the type of 'w' (line 53)
            w_204991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'w', False)
            # Getting the type of 'spread' (line 53)
            spread_204992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'spread', False)
            # Applying the binary operator '*' (line 53)
            result_mul_204993 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 21), '*', w_204991, spread_204992)
            
            # Obtaining the member 'sum' of a type (line 53)
            sum_204994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 21), result_mul_204993, 'sum')
            # Calling sum(args, kwargs) (line 53)
            sum_call_result_204996 = invoke(stypy.reporting.localization.Localization(__file__, 53, 21), sum_204994, *[], **kwargs_204995)
            
            # Applying the 'usub' unary operator (line 53)
            result___neg___204997 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 19), 'usub', sum_call_result_204996)
            
            # Assigning a type to the variable 'stypy_return_type' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'stypy_return_type', result___neg___204997)
            
            # ################# End of 'f(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'f' in the type store
            # Getting the type of 'stypy_return_type' (line 52)
            stypy_return_type_204998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_204998)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'f'
            return stypy_return_type_204998

        # Assigning a type to the variable 'f' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'f', f)

        @norecursion
        def c1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'c1'
            module_type_store = module_type_store.open_function_context('c1', 55, 8, False)
            
            # Passed parameters checking function
            c1.stypy_localization = localization
            c1.stypy_type_of_self = None
            c1.stypy_type_store = module_type_store
            c1.stypy_function_name = 'c1'
            c1.stypy_param_names_list = ['w']
            c1.stypy_varargs_param_name = None
            c1.stypy_kwargs_param_name = None
            c1.stypy_call_defaults = defaults
            c1.stypy_call_varargs = varargs
            c1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'c1', ['w'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'c1', localization, ['w'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'c1(...)' code ##################

            int_204999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'int')
            
            # Call to sum(...): (line 56)
            # Processing the call keyword arguments (line 56)
            kwargs_205008 = {}
            
            # Call to abs(...): (line 56)
            # Processing the call arguments (line 56)
            
            # Call to p(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'w' (line 56)
            w_205002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 31), 'w', False)
            # Processing the call keyword arguments (line 56)
            kwargs_205003 = {}
            # Getting the type of 'p' (line 56)
            p_205001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'p', False)
            # Calling p(args, kwargs) (line 56)
            p_call_result_205004 = invoke(stypy.reporting.localization.Localization(__file__, 56, 29), p_205001, *[w_205002], **kwargs_205003)
            
            # Processing the call keyword arguments (line 56)
            kwargs_205005 = {}
            # Getting the type of 'abs' (line 56)
            abs_205000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'abs', False)
            # Calling abs(args, kwargs) (line 56)
            abs_call_result_205006 = invoke(stypy.reporting.localization.Localization(__file__, 56, 25), abs_205000, *[p_call_result_205004], **kwargs_205005)
            
            # Obtaining the member 'sum' of a type (line 56)
            sum_205007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 25), abs_call_result_205006, 'sum')
            # Calling sum(args, kwargs) (line 56)
            sum_call_result_205009 = invoke(stypy.reporting.localization.Localization(__file__, 56, 25), sum_205007, *[], **kwargs_205008)
            
            # Applying the binary operator '-' (line 56)
            result_sub_205010 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 19), '-', int_204999, sum_call_result_205009)
            
            # Assigning a type to the variable 'stypy_return_type' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'stypy_return_type', result_sub_205010)
            
            # ################# End of 'c1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'c1' in the type store
            # Getting the type of 'stypy_return_type' (line 55)
            stypy_return_type_205011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205011)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'c1'
            return stypy_return_type_205011

        # Assigning a type to the variable 'c1' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'c1', c1)

        @norecursion
        def c2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'c2'
            module_type_store = module_type_store.open_function_context('c2', 58, 8, False)
            
            # Passed parameters checking function
            c2.stypy_localization = localization
            c2.stypy_type_of_self = None
            c2.stypy_type_store = module_type_store
            c2.stypy_function_name = 'c2'
            c2.stypy_param_names_list = ['w']
            c2.stypy_varargs_param_name = None
            c2.stypy_kwargs_param_name = None
            c2.stypy_call_defaults = defaults
            c2.stypy_call_varargs = varargs
            c2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'c2', ['w'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'c2', localization, ['w'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'c2(...)' code ##################

            int_205012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'int')
            
            # Call to abs(...): (line 59)
            # Processing the call arguments (line 59)
            
            # Call to sum(...): (line 59)
            # Processing the call keyword arguments (line 59)
            kwargs_205019 = {}
            
            # Call to p(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'w' (line 59)
            w_205015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'w', False)
            # Processing the call keyword arguments (line 59)
            kwargs_205016 = {}
            # Getting the type of 'p' (line 59)
            p_205014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'p', False)
            # Calling p(args, kwargs) (line 59)
            p_call_result_205017 = invoke(stypy.reporting.localization.Localization(__file__, 59, 27), p_205014, *[w_205015], **kwargs_205016)
            
            # Obtaining the member 'sum' of a type (line 59)
            sum_205018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 27), p_call_result_205017, 'sum')
            # Calling sum(args, kwargs) (line 59)
            sum_call_result_205020 = invoke(stypy.reporting.localization.Localization(__file__, 59, 27), sum_205018, *[], **kwargs_205019)
            
            # Processing the call keyword arguments (line 59)
            kwargs_205021 = {}
            # Getting the type of 'abs' (line 59)
            abs_205013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'abs', False)
            # Calling abs(args, kwargs) (line 59)
            abs_call_result_205022 = invoke(stypy.reporting.localization.Localization(__file__, 59, 23), abs_205013, *[sum_call_result_205020], **kwargs_205021)
            
            # Applying the binary operator '-' (line 59)
            result_sub_205023 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 19), '-', int_205012, abs_call_result_205022)
            
            # Assigning a type to the variable 'stypy_return_type' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'stypy_return_type', result_sub_205023)
            
            # ################# End of 'c2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'c2' in the type store
            # Getting the type of 'stypy_return_type' (line 58)
            stypy_return_type_205024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205024)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'c2'
            return stypy_return_type_205024

        # Assigning a type to the variable 'c2' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'c2', c2)

        @norecursion
        def c3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'c3'
            module_type_store = module_type_store.open_function_context('c3', 61, 8, False)
            
            # Passed parameters checking function
            c3.stypy_localization = localization
            c3.stypy_type_of_self = None
            c3.stypy_type_store = module_type_store
            c3.stypy_function_name = 'c3'
            c3.stypy_param_names_list = ['w']
            c3.stypy_varargs_param_name = None
            c3.stypy_kwargs_param_name = None
            c3.stypy_call_defaults = defaults
            c3.stypy_call_varargs = varargs
            c3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'c3', ['w'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'c3', localization, ['w'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'c3(...)' code ##################

            int_205025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'int')
            
            # Call to max(...): (line 62)
            # Processing the call keyword arguments (line 62)
            kwargs_205034 = {}
            
            # Call to abs(...): (line 62)
            # Processing the call arguments (line 62)
            
            # Call to p(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'w' (line 62)
            w_205028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 29), 'w', False)
            # Processing the call keyword arguments (line 62)
            kwargs_205029 = {}
            # Getting the type of 'p' (line 62)
            p_205027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'p', False)
            # Calling p(args, kwargs) (line 62)
            p_call_result_205030 = invoke(stypy.reporting.localization.Localization(__file__, 62, 27), p_205027, *[w_205028], **kwargs_205029)
            
            # Processing the call keyword arguments (line 62)
            kwargs_205031 = {}
            # Getting the type of 'abs' (line 62)
            abs_205026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'abs', False)
            # Calling abs(args, kwargs) (line 62)
            abs_call_result_205032 = invoke(stypy.reporting.localization.Localization(__file__, 62, 23), abs_205026, *[p_call_result_205030], **kwargs_205031)
            
            # Obtaining the member 'max' of a type (line 62)
            max_205033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 23), abs_call_result_205032, 'max')
            # Calling max(args, kwargs) (line 62)
            max_call_result_205035 = invoke(stypy.reporting.localization.Localization(__file__, 62, 23), max_205033, *[], **kwargs_205034)
            
            # Applying the binary operator '-' (line 62)
            result_sub_205036 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 19), '-', int_205025, max_call_result_205035)
            
            # Assigning a type to the variable 'stypy_return_type' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type', result_sub_205036)
            
            # ################# End of 'c3(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'c3' in the type store
            # Getting the type of 'stypy_return_type' (line 61)
            stypy_return_type_205037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205037)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'c3'
            return stypy_return_type_205037

        # Assigning a type to the variable 'c3' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'c3', c3)
        
        # Assigning a Tuple to a Name (line 64):
        
        # Obtaining an instance of the builtin type 'tuple' (line 64)
        tuple_205038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 64)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'dict' (line 64)
        dict_205039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 64)
        # Adding element type (key, value) (line 64)
        str_205040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 17), 'str', 'type')
        str_205041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_205039, (str_205040, str_205041))
        # Adding element type (key, value) (line 64)
        str_205042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 33), 'str', 'fun')
        # Getting the type of 'c1' (line 64)
        c1_205043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'c1')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_205039, (str_205042, c1_205043))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), tuple_205038, dict_205039)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'dict' (line 65)
        dict_205044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 65)
        # Adding element type (key, value) (line 65)
        str_205045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 17), 'str', 'type')
        str_205046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 16), dict_205044, (str_205045, str_205046))
        # Adding element type (key, value) (line 65)
        str_205047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'str', 'fun')
        # Getting the type of 'c2' (line 65)
        c2_205048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'c2')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 16), dict_205044, (str_205047, c2_205048))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), tuple_205038, dict_205044)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'dict' (line 66)
        dict_205049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 66)
        # Adding element type (key, value) (line 66)
        str_205050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 17), 'str', 'type')
        str_205051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 16), dict_205049, (str_205050, str_205051))
        # Adding element type (key, value) (line 66)
        str_205052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 33), 'str', 'fun')
        # Getting the type of 'c3' (line 66)
        c3_205053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 40), 'c3')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 16), dict_205049, (str_205052, c3_205053))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), tuple_205038, dict_205049)
        
        # Assigning a type to the variable 'cons' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'cons', tuple_205038)
        
        # Assigning a Call to a Name (line 67):
        
        # Call to zeros(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Obtaining an instance of the builtin type 'tuple' (line 67)
        tuple_205056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 67)
        # Adding element type (line 67)
        int_205057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 23), tuple_205056, int_205057)
        # Adding element type (line 67)
        int_205058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 23), tuple_205056, int_205058)
        
        # Processing the call keyword arguments (line 67)
        kwargs_205059 = {}
        # Getting the type of 'np' (line 67)
        np_205054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'np', False)
        # Obtaining the member 'zeros' of a type (line 67)
        zeros_205055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 13), np_205054, 'zeros')
        # Calling zeros(args, kwargs) (line 67)
        zeros_call_result_205060 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), zeros_205055, *[tuple_205056], **kwargs_205059)
        
        # Assigning a type to the variable 'w0' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'w0', zeros_call_result_205060)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to minimize(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'f' (line 68)
        f_205062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'f', False)
        # Getting the type of 'w0' (line 68)
        w0_205063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'w0', False)
        # Processing the call keyword arguments (line 68)
        str_205064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 37), 'str', 'cobyla')
        keyword_205065 = str_205064
        # Getting the type of 'cons' (line 68)
        cons_205066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 59), 'cons', False)
        keyword_205067 = cons_205066
        
        # Obtaining an instance of the builtin type 'dict' (line 69)
        dict_205068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 69)
        # Adding element type (key, value) (line 69)
        str_205069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'str', 'catol')
        float_205070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 41), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), dict_205068, (str_205069, float_205070))
        
        keyword_205071 = dict_205068
        kwargs_205072 = {'options': keyword_205071, 'method': keyword_205065, 'constraints': keyword_205067}
        # Getting the type of 'minimize' (line 68)
        minimize_205061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 68)
        minimize_call_result_205073 = invoke(stypy.reporting.localization.Localization(__file__, 68, 14), minimize_205061, *[f_205062, w0_205063], **kwargs_205072)
        
        # Assigning a type to the variable 'sol' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'sol', minimize_call_result_205073)
        
        # Call to assert_(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Getting the type of 'sol' (line 70)
        sol_205075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'sol', False)
        # Obtaining the member 'maxcv' of a type (line 70)
        maxcv_205076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), sol_205075, 'maxcv')
        float_205077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'float')
        # Applying the binary operator '>' (line 70)
        result_gt_205078 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 16), '>', maxcv_205076, float_205077)
        
        # Processing the call keyword arguments (line 70)
        kwargs_205079 = {}
        # Getting the type of 'assert_' (line 70)
        assert__205074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 70)
        assert__call_result_205080 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assert__205074, *[result_gt_205078], **kwargs_205079)
        
        
        # Call to assert_(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Getting the type of 'sol' (line 71)
        sol_205082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'sol', False)
        # Obtaining the member 'success' of a type (line 71)
        success_205083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), sol_205082, 'success')
        # Applying the 'not' unary operator (line 71)
        result_not__205084 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 16), 'not', success_205083)
        
        # Processing the call keyword arguments (line 71)
        kwargs_205085 = {}
        # Getting the type of 'assert_' (line 71)
        assert__205081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 71)
        assert__call_result_205086 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert__205081, *[result_not__205084], **kwargs_205085)
        
        
        # ################# End of 'test_minimize_constraint_violation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_constraint_violation' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_205087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205087)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_constraint_violation'
        return stypy_return_type_205087


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 0, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCobyla.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCobyla' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'TestCobyla', TestCobyla)

@norecursion
def test_vector_constraints(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_vector_constraints'
    module_type_store = module_type_store.open_function_context('test_vector_constraints', 74, 0, False)
    
    # Passed parameters checking function
    test_vector_constraints.stypy_localization = localization
    test_vector_constraints.stypy_type_of_self = None
    test_vector_constraints.stypy_type_store = module_type_store
    test_vector_constraints.stypy_function_name = 'test_vector_constraints'
    test_vector_constraints.stypy_param_names_list = []
    test_vector_constraints.stypy_varargs_param_name = None
    test_vector_constraints.stypy_kwargs_param_name = None
    test_vector_constraints.stypy_call_defaults = defaults
    test_vector_constraints.stypy_call_varargs = varargs
    test_vector_constraints.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_vector_constraints', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_vector_constraints', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_vector_constraints(...)' code ##################


    @norecursion
    def fun(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun'
        module_type_store = module_type_store.open_function_context('fun', 77, 4, False)
        
        # Passed parameters checking function
        fun.stypy_localization = localization
        fun.stypy_type_of_self = None
        fun.stypy_type_store = module_type_store
        fun.stypy_function_name = 'fun'
        fun.stypy_param_names_list = ['x']
        fun.stypy_varargs_param_name = None
        fun.stypy_kwargs_param_name = None
        fun.stypy_call_defaults = defaults
        fun.stypy_call_varargs = varargs
        fun.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun(...)' code ##################

        
        # Obtaining the type of the subscript
        int_205088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 18), 'int')
        # Getting the type of 'x' (line 78)
        x_205089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'x')
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___205090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), x_205089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_205091 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), getitem___205090, int_205088)
        
        int_205092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
        # Applying the binary operator '-' (line 78)
        result_sub_205093 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '-', subscript_call_result_205091, int_205092)
        
        int_205094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'int')
        # Applying the binary operator '**' (line 78)
        result_pow_205095 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), '**', result_sub_205093, int_205094)
        
        
        # Obtaining the type of the subscript
        int_205096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'int')
        # Getting the type of 'x' (line 78)
        x_205097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 32), 'x')
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___205098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 32), x_205097, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_205099 = invoke(stypy.reporting.localization.Localization(__file__, 78, 32), getitem___205098, int_205096)
        
        float_205100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 39), 'float')
        # Applying the binary operator '-' (line 78)
        result_sub_205101 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 32), '-', subscript_call_result_205099, float_205100)
        
        int_205102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 45), 'int')
        # Applying the binary operator '**' (line 78)
        result_pow_205103 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 31), '**', result_sub_205101, int_205102)
        
        # Applying the binary operator '+' (line 78)
        result_add_205104 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), '+', result_pow_205095, result_pow_205103)
        
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', result_add_205104)
        
        # ################# End of 'fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_205105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205105)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun'
        return stypy_return_type_205105

    # Assigning a type to the variable 'fun' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'fun', fun)

    @norecursion
    def fmin(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fmin'
        module_type_store = module_type_store.open_function_context('fmin', 80, 4, False)
        
        # Passed parameters checking function
        fmin.stypy_localization = localization
        fmin.stypy_type_of_self = None
        fmin.stypy_type_store = module_type_store
        fmin.stypy_function_name = 'fmin'
        fmin.stypy_param_names_list = ['x']
        fmin.stypy_varargs_param_name = None
        fmin.stypy_kwargs_param_name = None
        fmin.stypy_call_defaults = defaults
        fmin.stypy_call_varargs = varargs
        fmin.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fmin', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fmin', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fmin(...)' code ##################

        
        # Call to fun(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'x' (line 81)
        x_205107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'x', False)
        # Processing the call keyword arguments (line 81)
        kwargs_205108 = {}
        # Getting the type of 'fun' (line 81)
        fun_205106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'fun', False)
        # Calling fun(args, kwargs) (line 81)
        fun_call_result_205109 = invoke(stypy.reporting.localization.Localization(__file__, 81, 15), fun_205106, *[x_205107], **kwargs_205108)
        
        int_205110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'int')
        # Applying the binary operator '-' (line 81)
        result_sub_205111 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), '-', fun_call_result_205109, int_205110)
        
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', result_sub_205111)
        
        # ################# End of 'fmin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fmin' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_205112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fmin'
        return stypy_return_type_205112

    # Assigning a type to the variable 'fmin' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'fmin', fmin)

    @norecursion
    def cons1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cons1'
        module_type_store = module_type_store.open_function_context('cons1', 83, 4, False)
        
        # Passed parameters checking function
        cons1.stypy_localization = localization
        cons1.stypy_type_of_self = None
        cons1.stypy_type_store = module_type_store
        cons1.stypy_function_name = 'cons1'
        cons1.stypy_param_names_list = ['x']
        cons1.stypy_varargs_param_name = None
        cons1.stypy_kwargs_param_name = None
        cons1.stypy_call_defaults = defaults
        cons1.stypy_call_varargs = varargs
        cons1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'cons1', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cons1', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cons1(...)' code ##################

        
        # Assigning a Call to a Name (line 84):
        
        # Call to array(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_205115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_205116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_205117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_205116, int_205117)
        # Adding element type (line 84)
        int_205118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_205116, int_205118)
        # Adding element type (line 84)
        int_205119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_205116, int_205119)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), list_205115, list_205116)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_205120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_205121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 34), list_205120, int_205121)
        # Adding element type (line 84)
        int_205122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 34), list_205120, int_205122)
        # Adding element type (line 84)
        int_205123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 34), list_205120, int_205123)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), list_205115, list_205120)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_205124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_205125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 47), list_205124, int_205125)
        # Adding element type (line 84)
        int_205126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 47), list_205124, int_205126)
        # Adding element type (line 84)
        int_205127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 47), list_205124, int_205127)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), list_205115, list_205124)
        
        # Processing the call keyword arguments (line 84)
        kwargs_205128 = {}
        # Getting the type of 'np' (line 84)
        np_205113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 84)
        array_205114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), np_205113, 'array')
        # Calling array(args, kwargs) (line 84)
        array_call_result_205129 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), array_205114, *[list_205115], **kwargs_205128)
        
        # Assigning a type to the variable 'a' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'a', array_call_result_205129)
        
        # Call to array(...): (line 85)
        # Processing the call arguments (line 85)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to len(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'a' (line 86)
        a_205164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 52), 'a', False)
        # Processing the call keyword arguments (line 86)
        kwargs_205165 = {}
        # Getting the type of 'len' (line 86)
        len_205163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 48), 'len', False)
        # Calling len(args, kwargs) (line 86)
        len_call_result_205166 = invoke(stypy.reporting.localization.Localization(__file__, 86, 48), len_205163, *[a_205164], **kwargs_205165)
        
        # Processing the call keyword arguments (line 86)
        kwargs_205167 = {}
        # Getting the type of 'range' (line 86)
        range_205162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'range', False)
        # Calling range(args, kwargs) (line 86)
        range_call_result_205168 = invoke(stypy.reporting.localization.Localization(__file__, 86, 42), range_205162, *[len_call_result_205166], **kwargs_205167)
        
        comprehension_205169 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 25), range_call_result_205168)
        # Assigning a type to the variable 'i' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'i', comprehension_205169)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_205132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        # Getting the type of 'i' (line 85)
        i_205133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 27), tuple_205132, i_205133)
        # Adding element type (line 85)
        int_205134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 27), tuple_205132, int_205134)
        
        # Getting the type of 'a' (line 85)
        a_205135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___205136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 25), a_205135, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_205137 = invoke(stypy.reporting.localization.Localization(__file__, 85, 25), getitem___205136, tuple_205132)
        
        
        # Obtaining the type of the subscript
        int_205138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'int')
        # Getting the type of 'x' (line 85)
        x_205139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 35), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___205140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 35), x_205139, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_205141 = invoke(stypy.reporting.localization.Localization(__file__, 85, 35), getitem___205140, int_205138)
        
        # Applying the binary operator '*' (line 85)
        result_mul_205142 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 25), '*', subscript_call_result_205137, subscript_call_result_205141)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_205143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        # Getting the type of 'i' (line 85)
        i_205144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 44), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 44), tuple_205143, i_205144)
        # Adding element type (line 85)
        int_205145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 44), tuple_205143, int_205145)
        
        # Getting the type of 'a' (line 85)
        a_205146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 42), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___205147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 42), a_205146, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_205148 = invoke(stypy.reporting.localization.Localization(__file__, 85, 42), getitem___205147, tuple_205143)
        
        
        # Obtaining the type of the subscript
        int_205149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 54), 'int')
        # Getting the type of 'x' (line 85)
        x_205150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 52), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___205151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 52), x_205150, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_205152 = invoke(stypy.reporting.localization.Localization(__file__, 85, 52), getitem___205151, int_205149)
        
        # Applying the binary operator '*' (line 85)
        result_mul_205153 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 42), '*', subscript_call_result_205148, subscript_call_result_205152)
        
        # Applying the binary operator '+' (line 85)
        result_add_205154 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 25), '+', result_mul_205142, result_mul_205153)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_205155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        # Getting the type of 'i' (line 86)
        i_205156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 27), tuple_205155, i_205156)
        # Adding element type (line 86)
        int_205157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 27), tuple_205155, int_205157)
        
        # Getting the type of 'a' (line 86)
        a_205158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___205159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 25), a_205158, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_205160 = invoke(stypy.reporting.localization.Localization(__file__, 86, 25), getitem___205159, tuple_205155)
        
        # Applying the binary operator '+' (line 85)
        result_add_205161 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 57), '+', result_add_205154, subscript_call_result_205160)
        
        list_205170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 25), list_205170, result_add_205161)
        # Processing the call keyword arguments (line 85)
        kwargs_205171 = {}
        # Getting the type of 'np' (line 85)
        np_205130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 85)
        array_205131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), np_205130, 'array')
        # Calling array(args, kwargs) (line 85)
        array_call_result_205172 = invoke(stypy.reporting.localization.Localization(__file__, 85, 15), array_205131, *[list_205170], **kwargs_205171)
        
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', array_call_result_205172)
        
        # ################# End of 'cons1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cons1' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_205173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cons1'
        return stypy_return_type_205173

    # Assigning a type to the variable 'cons1' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'cons1', cons1)

    @norecursion
    def cons2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cons2'
        module_type_store = module_type_store.open_function_context('cons2', 88, 4, False)
        
        # Passed parameters checking function
        cons2.stypy_localization = localization
        cons2.stypy_type_of_self = None
        cons2.stypy_type_store = module_type_store
        cons2.stypy_function_name = 'cons2'
        cons2.stypy_param_names_list = ['x']
        cons2.stypy_varargs_param_name = None
        cons2.stypy_kwargs_param_name = None
        cons2.stypy_call_defaults = defaults
        cons2.stypy_call_varargs = varargs
        cons2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'cons2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cons2', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cons2(...)' code ##################

        # Getting the type of 'x' (line 89)
        x_205174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'stypy_return_type', x_205174)
        
        # ################# End of 'cons2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cons2' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_205175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cons2'
        return stypy_return_type_205175

    # Assigning a type to the variable 'cons2' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'cons2', cons2)
    
    # Assigning a Call to a Name (line 91):
    
    # Call to array(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Obtaining an instance of the builtin type 'list' (line 91)
    list_205178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 91)
    # Adding element type (line 91)
    int_205179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 18), list_205178, int_205179)
    # Adding element type (line 91)
    int_205180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 18), list_205178, int_205180)
    
    # Processing the call keyword arguments (line 91)
    kwargs_205181 = {}
    # Getting the type of 'np' (line 91)
    np_205176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 91)
    array_205177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 9), np_205176, 'array')
    # Calling array(args, kwargs) (line 91)
    array_call_result_205182 = invoke(stypy.reporting.localization.Localization(__file__, 91, 9), array_205177, *[list_205178], **kwargs_205181)
    
    # Assigning a type to the variable 'x0' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'x0', array_call_result_205182)
    
    # Assigning a List to a Name (line 92):
    
    # Obtaining an instance of the builtin type 'list' (line 92)
    list_205183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 92)
    # Adding element type (line 92)
    # Getting the type of 'fun' (line 92)
    fun_205184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'fun')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 16), list_205183, fun_205184)
    # Adding element type (line 92)
    # Getting the type of 'cons1' (line 92)
    cons1_205185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'cons1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 16), list_205183, cons1_205185)
    # Adding element type (line 92)
    # Getting the type of 'cons2' (line 92)
    cons2_205186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'cons2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 16), list_205183, cons2_205186)
    
    # Assigning a type to the variable 'cons_list' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'cons_list', list_205183)
    
    # Assigning a List to a Name (line 94):
    
    # Obtaining an instance of the builtin type 'list' (line 94)
    list_205187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 94)
    # Adding element type (line 94)
    float_205188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 11), list_205187, float_205188)
    # Adding element type (line 94)
    float_205189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 11), list_205187, float_205189)
    
    # Assigning a type to the variable 'xsol' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'xsol', list_205187)
    
    # Assigning a Num to a Name (line 95):
    float_205190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'float')
    # Assigning a type to the variable 'fsol' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'fsol', float_205190)
    
    # Assigning a Call to a Name (line 98):
    
    # Call to fmin_cobyla(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'fun' (line 98)
    fun_205192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'fun', False)
    # Getting the type of 'x0' (line 98)
    x0_205193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'x0', False)
    # Getting the type of 'cons_list' (line 98)
    cons_list_205194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'cons_list', False)
    # Processing the call keyword arguments (line 98)
    float_205195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 49), 'float')
    keyword_205196 = float_205195
    kwargs_205197 = {'rhoend': keyword_205196}
    # Getting the type of 'fmin_cobyla' (line 98)
    fmin_cobyla_205191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 10), 'fmin_cobyla', False)
    # Calling fmin_cobyla(args, kwargs) (line 98)
    fmin_cobyla_call_result_205198 = invoke(stypy.reporting.localization.Localization(__file__, 98, 10), fmin_cobyla_205191, *[fun_205192, x0_205193, cons_list_205194], **kwargs_205197)
    
    # Assigning a type to the variable 'sol' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'sol', fmin_cobyla_call_result_205198)
    
    # Call to assert_allclose(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'sol' (line 99)
    sol_205200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'sol', False)
    # Getting the type of 'xsol' (line 99)
    xsol_205201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'xsol', False)
    # Processing the call keyword arguments (line 99)
    float_205202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 36), 'float')
    keyword_205203 = float_205202
    kwargs_205204 = {'atol': keyword_205203}
    # Getting the type of 'assert_allclose' (line 99)
    assert_allclose_205199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 99)
    assert_allclose_call_result_205205 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), assert_allclose_205199, *[sol_205200, xsol_205201], **kwargs_205204)
    
    
    # Assigning a Call to a Name (line 101):
    
    # Call to fmin_cobyla(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'fun' (line 101)
    fun_205207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'fun', False)
    # Getting the type of 'x0' (line 101)
    x0_205208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'x0', False)
    # Getting the type of 'fmin' (line 101)
    fmin_205209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'fmin', False)
    # Processing the call keyword arguments (line 101)
    float_205210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 44), 'float')
    keyword_205211 = float_205210
    kwargs_205212 = {'rhoend': keyword_205211}
    # Getting the type of 'fmin_cobyla' (line 101)
    fmin_cobyla_205206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), 'fmin_cobyla', False)
    # Calling fmin_cobyla(args, kwargs) (line 101)
    fmin_cobyla_call_result_205213 = invoke(stypy.reporting.localization.Localization(__file__, 101, 10), fmin_cobyla_205206, *[fun_205207, x0_205208, fmin_205209], **kwargs_205212)
    
    # Assigning a type to the variable 'sol' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'sol', fmin_cobyla_call_result_205213)
    
    # Call to assert_allclose(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Call to fun(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'sol' (line 102)
    sol_205216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'sol', False)
    # Processing the call keyword arguments (line 102)
    kwargs_205217 = {}
    # Getting the type of 'fun' (line 102)
    fun_205215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'fun', False)
    # Calling fun(args, kwargs) (line 102)
    fun_call_result_205218 = invoke(stypy.reporting.localization.Localization(__file__, 102, 20), fun_205215, *[sol_205216], **kwargs_205217)
    
    int_205219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 30), 'int')
    # Processing the call keyword arguments (line 102)
    float_205220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 38), 'float')
    keyword_205221 = float_205220
    kwargs_205222 = {'atol': keyword_205221}
    # Getting the type of 'assert_allclose' (line 102)
    assert_allclose_205214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 102)
    assert_allclose_call_result_205223 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), assert_allclose_205214, *[fun_call_result_205218, int_205219], **kwargs_205222)
    
    
    # Assigning a ListComp to a Name (line 105):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'cons_list' (line 105)
    cons_list_205229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 61), 'cons_list')
    comprehension_205230 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), cons_list_205229)
    # Assigning a type to the variable 'cons' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'cons', comprehension_205230)
    
    # Obtaining an instance of the builtin type 'dict' (line 105)
    dict_205224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 105)
    # Adding element type (key, value) (line 105)
    str_205225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'str', 'type')
    str_205226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'str', 'ineq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), dict_205224, (str_205225, str_205226))
    # Adding element type (key, value) (line 105)
    str_205227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'str', 'fun')
    # Getting the type of 'cons' (line 105)
    cons_205228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 43), 'cons')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), dict_205224, (str_205227, cons_205228))
    
    list_205231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_205231, dict_205224)
    # Assigning a type to the variable 'constraints' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'constraints', list_205231)
    
    # Assigning a Call to a Name (line 106):
    
    # Call to minimize(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'fun' (line 106)
    fun_205233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'fun', False)
    # Getting the type of 'x0' (line 106)
    x0_205234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'x0', False)
    # Processing the call keyword arguments (line 106)
    # Getting the type of 'constraints' (line 106)
    constraints_205235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'constraints', False)
    keyword_205236 = constraints_205235
    float_205237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 57), 'float')
    keyword_205238 = float_205237
    kwargs_205239 = {'tol': keyword_205238, 'constraints': keyword_205236}
    # Getting the type of 'minimize' (line 106)
    minimize_205232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 10), 'minimize', False)
    # Calling minimize(args, kwargs) (line 106)
    minimize_call_result_205240 = invoke(stypy.reporting.localization.Localization(__file__, 106, 10), minimize_205232, *[fun_205233, x0_205234], **kwargs_205239)
    
    # Assigning a type to the variable 'sol' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'sol', minimize_call_result_205240)
    
    # Call to assert_allclose(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'sol' (line 107)
    sol_205242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 20), 'sol', False)
    # Obtaining the member 'x' of a type (line 107)
    x_205243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 20), sol_205242, 'x')
    # Getting the type of 'xsol' (line 107)
    xsol_205244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'xsol', False)
    # Processing the call keyword arguments (line 107)
    float_205245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 38), 'float')
    keyword_205246 = float_205245
    kwargs_205247 = {'atol': keyword_205246}
    # Getting the type of 'assert_allclose' (line 107)
    assert_allclose_205241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 107)
    assert_allclose_call_result_205248 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), assert_allclose_205241, *[x_205243, xsol_205244], **kwargs_205247)
    
    
    # Call to assert_(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'sol' (line 108)
    sol_205250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'sol', False)
    # Obtaining the member 'success' of a type (line 108)
    success_205251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), sol_205250, 'success')
    # Getting the type of 'sol' (line 108)
    sol_205252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'sol', False)
    # Obtaining the member 'message' of a type (line 108)
    message_205253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 25), sol_205252, 'message')
    # Processing the call keyword arguments (line 108)
    kwargs_205254 = {}
    # Getting the type of 'assert_' (line 108)
    assert__205249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 108)
    assert__call_result_205255 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), assert__205249, *[success_205251, message_205253], **kwargs_205254)
    
    
    # Call to assert_allclose(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'sol' (line 109)
    sol_205257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'sol', False)
    # Obtaining the member 'fun' of a type (line 109)
    fun_205258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 20), sol_205257, 'fun')
    # Getting the type of 'fsol' (line 109)
    fsol_205259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 29), 'fsol', False)
    # Processing the call keyword arguments (line 109)
    float_205260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 40), 'float')
    keyword_205261 = float_205260
    kwargs_205262 = {'atol': keyword_205261}
    # Getting the type of 'assert_allclose' (line 109)
    assert_allclose_205256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 109)
    assert_allclose_call_result_205263 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), assert_allclose_205256, *[fun_205258, fsol_205259], **kwargs_205262)
    
    
    # Assigning a Dict to a Name (line 111):
    
    # Obtaining an instance of the builtin type 'dict' (line 111)
    dict_205264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 111)
    # Adding element type (key, value) (line 111)
    str_205265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 19), 'str', 'type')
    str_205266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 27), 'str', 'ineq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 18), dict_205264, (str_205265, str_205266))
    # Adding element type (key, value) (line 111)
    str_205267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'str', 'fun')
    # Getting the type of 'fmin' (line 111)
    fmin_205268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'fmin')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 18), dict_205264, (str_205267, fmin_205268))
    
    # Assigning a type to the variable 'constraints' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'constraints', dict_205264)
    
    # Assigning a Call to a Name (line 112):
    
    # Call to minimize(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'fun' (line 112)
    fun_205270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'fun', False)
    # Getting the type of 'x0' (line 112)
    x0_205271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'x0', False)
    # Processing the call keyword arguments (line 112)
    # Getting the type of 'constraints' (line 112)
    constraints_205272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'constraints', False)
    keyword_205273 = constraints_205272
    float_205274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 57), 'float')
    keyword_205275 = float_205274
    kwargs_205276 = {'tol': keyword_205275, 'constraints': keyword_205273}
    # Getting the type of 'minimize' (line 112)
    minimize_205269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 10), 'minimize', False)
    # Calling minimize(args, kwargs) (line 112)
    minimize_call_result_205277 = invoke(stypy.reporting.localization.Localization(__file__, 112, 10), minimize_205269, *[fun_205270, x0_205271], **kwargs_205276)
    
    # Assigning a type to the variable 'sol' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'sol', minimize_call_result_205277)
    
    # Call to assert_allclose(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'sol' (line 113)
    sol_205279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'sol', False)
    # Obtaining the member 'fun' of a type (line 113)
    fun_205280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), sol_205279, 'fun')
    int_205281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'int')
    # Processing the call keyword arguments (line 113)
    float_205282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'float')
    keyword_205283 = float_205282
    kwargs_205284 = {'atol': keyword_205283}
    # Getting the type of 'assert_allclose' (line 113)
    assert_allclose_205278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 113)
    assert_allclose_call_result_205285 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), assert_allclose_205278, *[fun_205280, int_205281], **kwargs_205284)
    
    
    # ################# End of 'test_vector_constraints(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_vector_constraints' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_205286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205286)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_vector_constraints'
    return stypy_return_type_205286

# Assigning a type to the variable 'test_vector_constraints' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'test_vector_constraints', test_vector_constraints)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
