
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from math import sqrt, exp, sin, cos
4: 
5: from numpy.testing import (assert_warns, assert_, 
6:                            assert_allclose,
7:                            assert_equal)
8: from numpy import finfo
9: 
10: from scipy.optimize import zeros as cc
11: from scipy.optimize import zeros
12: 
13: # Import testing parameters
14: from scipy.optimize._tstutils import functions, fstrings
15: 
16: 
17: class TestBasic(object):
18:     def run_check(self, method, name):
19:         a = .5
20:         b = sqrt(3)
21:         xtol = 4*finfo(float).eps
22:         rtol = 4*finfo(float).eps
23:         for function, fname in zip(functions, fstrings):
24:             zero, r = method(function, a, b, xtol=xtol, rtol=rtol,
25:                              full_output=True)
26:             assert_(r.converged)
27:             assert_allclose(zero, 1.0, atol=xtol, rtol=rtol,
28:                 err_msg='method %s, function %s' % (name, fname))
29: 
30:     def test_bisect(self):
31:         self.run_check(cc.bisect, 'bisect')
32: 
33:     def test_ridder(self):
34:         self.run_check(cc.ridder, 'ridder')
35: 
36:     def test_brentq(self):
37:         self.run_check(cc.brentq, 'brentq')
38: 
39:     def test_brenth(self):
40:         self.run_check(cc.brenth, 'brenth')
41: 
42:     def test_newton(self):
43:         f1 = lambda x: x**2 - 2*x - 1
44:         f1_1 = lambda x: 2*x - 2
45:         f1_2 = lambda x: 2.0 + 0*x
46: 
47:         f2 = lambda x: exp(x) - cos(x)
48:         f2_1 = lambda x: exp(x) + sin(x)
49:         f2_2 = lambda x: exp(x) + cos(x)
50: 
51:         for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
52:             x = zeros.newton(f, 3, tol=1e-6)
53:             assert_allclose(f(x), 0, atol=1e-6)
54:             x = zeros.newton(f, 3, fprime=f_1, tol=1e-6)
55:             assert_allclose(f(x), 0, atol=1e-6)
56:             x = zeros.newton(f, 3, fprime=f_1, fprime2=f_2, tol=1e-6)
57:             assert_allclose(f(x), 0, atol=1e-6)
58: 
59:     def test_deriv_zero_warning(self):
60:         func = lambda x: x**2
61:         dfunc = lambda x: 2*x
62:         assert_warns(RuntimeWarning, cc.newton, func, 0.0, dfunc)
63: 
64: 
65: def test_gh_5555():
66:     root = 0.1
67: 
68:     def f(x):
69:         return x - root
70: 
71:     methods = [cc.bisect, cc.ridder]
72:     xtol = 4*finfo(float).eps
73:     rtol = 4*finfo(float).eps
74:     for method in methods:
75:         res = method(f, -1e8, 1e7, xtol=xtol, rtol=rtol)
76:         assert_allclose(root, res, atol=xtol, rtol=rtol,
77:                         err_msg='method %s' % method.__name__)
78: 
79: 
80: def test_gh_5557():
81:     # Show that without the changes in 5557 brentq and brenth might
82:     # only achieve a tolerance of 2*(xtol + rtol*|res|).
83: 
84:     # f linearly interpolates (0, -0.1), (0.5, -0.1), and (1,
85:     # 0.4). The important parts are that |f(0)| < |f(1)| (so that
86:     # brent takes 0 as the initial guess), |f(0)| < atol (so that
87:     # brent accepts 0 as the root), and that the exact root of f lies
88:     # more than atol away from 0 (so that brent doesn't achieve the
89:     # desired tolerance).
90:     def f(x):
91:         if x < 0.5:
92:             return -0.1
93:         else:
94:             return x - 0.6
95: 
96:     atol = 0.51
97:     rtol = 4*finfo(float).eps
98:     methods = [cc.brentq, cc.brenth]
99:     for method in methods:
100:         res = method(f, 0, 1, xtol=atol, rtol=rtol)
101:         assert_allclose(0.6, res, atol=atol, rtol=rtol)
102: 
103: 
104: class TestRootResults:
105:     def test_repr(self):
106:         r = zeros.RootResults(root=1.0,
107:                               iterations=44,
108:                               function_calls=46,
109:                               flag=0)
110:         expected_repr = ("      converged: True\n           flag: 'converged'"
111:                          "\n function_calls: 46\n     iterations: 44\n"
112:                          "           root: 1.0")
113:         assert_equal(repr(r), expected_repr)
114: 
115: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from math import sqrt, exp, sin, cos' statement (line 3)
try:
    from math import sqrt, exp, sin, cos

except:
    sqrt = UndefinedType
    exp = UndefinedType
    sin = UndefinedType
    cos = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'math', None, module_type_store, ['sqrt', 'exp', 'sin', 'cos'], [sqrt, exp, sin, cos])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_warns, assert_, assert_allclose, assert_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236410 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_236410) is not StypyTypeError):

    if (import_236410 != 'pyd_module'):
        __import__(import_236410)
        sys_modules_236411 = sys.modules[import_236410]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_236411.module_type_store, module_type_store, ['assert_warns', 'assert_', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_236411, sys_modules_236411.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_warns, assert_, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_warns', 'assert_', 'assert_allclose', 'assert_equal'], [assert_warns, assert_, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_236410)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy import finfo' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236412 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_236412) is not StypyTypeError):

    if (import_236412 != 'pyd_module'):
        __import__(import_236412)
        sys_modules_236413 = sys.modules[import_236412]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', sys_modules_236413.module_type_store, module_type_store, ['finfo'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_236413, sys_modules_236413.module_type_store, module_type_store)
    else:
        from numpy import finfo

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', None, module_type_store, ['finfo'], [finfo])

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_236412)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.optimize import cc' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236414 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize')

if (type(import_236414) is not StypyTypeError):

    if (import_236414 != 'pyd_module'):
        __import__(import_236414)
        sys_modules_236415 = sys.modules[import_236414]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize', sys_modules_236415.module_type_store, module_type_store, ['zeros'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_236415, sys_modules_236415.module_type_store, module_type_store)
    else:
        from scipy.optimize import zeros as cc

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize', None, module_type_store, ['zeros'], [cc])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize', import_236414)

# Adding an alias
module_type_store.add_alias('cc', 'zeros')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize import zeros' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236416 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize')

if (type(import_236416) is not StypyTypeError):

    if (import_236416 != 'pyd_module'):
        __import__(import_236416)
        sys_modules_236417 = sys.modules[import_236416]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', sys_modules_236417.module_type_store, module_type_store, ['zeros'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_236417, sys_modules_236417.module_type_store, module_type_store)
    else:
        from scipy.optimize import zeros

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', None, module_type_store, ['zeros'], [zeros])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', import_236416)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.optimize._tstutils import functions, fstrings' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236418 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.optimize._tstutils')

if (type(import_236418) is not StypyTypeError):

    if (import_236418 != 'pyd_module'):
        __import__(import_236418)
        sys_modules_236419 = sys.modules[import_236418]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.optimize._tstutils', sys_modules_236419.module_type_store, module_type_store, ['functions', 'fstrings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_236419, sys_modules_236419.module_type_store, module_type_store)
    else:
        from scipy.optimize._tstutils import functions, fstrings

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.optimize._tstutils', None, module_type_store, ['functions', 'fstrings'], [functions, fstrings])

else:
    # Assigning a type to the variable 'scipy.optimize._tstutils' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.optimize._tstutils', import_236418)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'TestBasic' class

class TestBasic(object, ):

    @norecursion
    def run_check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run_check'
        module_type_store = module_type_store.open_function_context('run_check', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasic.run_check.__dict__.__setitem__('stypy_localization', localization)
        TestBasic.run_check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasic.run_check.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasic.run_check.__dict__.__setitem__('stypy_function_name', 'TestBasic.run_check')
        TestBasic.run_check.__dict__.__setitem__('stypy_param_names_list', ['method', 'name'])
        TestBasic.run_check.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasic.run_check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasic.run_check.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasic.run_check.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasic.run_check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasic.run_check.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasic.run_check', ['method', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run_check', localization, ['method', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run_check(...)' code ##################

        
        # Assigning a Num to a Name (line 19):
        
        # Assigning a Num to a Name (line 19):
        float_236420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 12), 'float')
        # Assigning a type to the variable 'a' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'a', float_236420)
        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to sqrt(...): (line 20)
        # Processing the call arguments (line 20)
        int_236422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 17), 'int')
        # Processing the call keyword arguments (line 20)
        kwargs_236423 = {}
        # Getting the type of 'sqrt' (line 20)
        sqrt_236421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 20)
        sqrt_call_result_236424 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), sqrt_236421, *[int_236422], **kwargs_236423)
        
        # Assigning a type to the variable 'b' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'b', sqrt_call_result_236424)
        
        # Assigning a BinOp to a Name (line 21):
        
        # Assigning a BinOp to a Name (line 21):
        int_236425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'int')
        
        # Call to finfo(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'float' (line 21)
        float_236427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'float', False)
        # Processing the call keyword arguments (line 21)
        kwargs_236428 = {}
        # Getting the type of 'finfo' (line 21)
        finfo_236426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'finfo', False)
        # Calling finfo(args, kwargs) (line 21)
        finfo_call_result_236429 = invoke(stypy.reporting.localization.Localization(__file__, 21, 17), finfo_236426, *[float_236427], **kwargs_236428)
        
        # Obtaining the member 'eps' of a type (line 21)
        eps_236430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 17), finfo_call_result_236429, 'eps')
        # Applying the binary operator '*' (line 21)
        result_mul_236431 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 15), '*', int_236425, eps_236430)
        
        # Assigning a type to the variable 'xtol' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'xtol', result_mul_236431)
        
        # Assigning a BinOp to a Name (line 22):
        
        # Assigning a BinOp to a Name (line 22):
        int_236432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'int')
        
        # Call to finfo(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'float' (line 22)
        float_236434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'float', False)
        # Processing the call keyword arguments (line 22)
        kwargs_236435 = {}
        # Getting the type of 'finfo' (line 22)
        finfo_236433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'finfo', False)
        # Calling finfo(args, kwargs) (line 22)
        finfo_call_result_236436 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), finfo_236433, *[float_236434], **kwargs_236435)
        
        # Obtaining the member 'eps' of a type (line 22)
        eps_236437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), finfo_call_result_236436, 'eps')
        # Applying the binary operator '*' (line 22)
        result_mul_236438 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '*', int_236432, eps_236437)
        
        # Assigning a type to the variable 'rtol' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'rtol', result_mul_236438)
        
        
        # Call to zip(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'functions' (line 23)
        functions_236440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 35), 'functions', False)
        # Getting the type of 'fstrings' (line 23)
        fstrings_236441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 46), 'fstrings', False)
        # Processing the call keyword arguments (line 23)
        kwargs_236442 = {}
        # Getting the type of 'zip' (line 23)
        zip_236439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'zip', False)
        # Calling zip(args, kwargs) (line 23)
        zip_call_result_236443 = invoke(stypy.reporting.localization.Localization(__file__, 23, 31), zip_236439, *[functions_236440, fstrings_236441], **kwargs_236442)
        
        # Testing the type of a for loop iterable (line 23)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 23, 8), zip_call_result_236443)
        # Getting the type of the for loop variable (line 23)
        for_loop_var_236444 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 23, 8), zip_call_result_236443)
        # Assigning a type to the variable 'function' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'function', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), for_loop_var_236444))
        # Assigning a type to the variable 'fname' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'fname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), for_loop_var_236444))
        # SSA begins for a for statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 24):
        
        # Assigning a Subscript to a Name (line 24):
        
        # Obtaining the type of the subscript
        int_236445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        
        # Call to method(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'function' (line 24)
        function_236447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'function', False)
        # Getting the type of 'a' (line 24)
        a_236448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 39), 'a', False)
        # Getting the type of 'b' (line 24)
        b_236449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'b', False)
        # Processing the call keyword arguments (line 24)
        # Getting the type of 'xtol' (line 24)
        xtol_236450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 50), 'xtol', False)
        keyword_236451 = xtol_236450
        # Getting the type of 'rtol' (line 24)
        rtol_236452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 61), 'rtol', False)
        keyword_236453 = rtol_236452
        # Getting the type of 'True' (line 25)
        True_236454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 41), 'True', False)
        keyword_236455 = True_236454
        kwargs_236456 = {'xtol': keyword_236451, 'rtol': keyword_236453, 'full_output': keyword_236455}
        # Getting the type of 'method' (line 24)
        method_236446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'method', False)
        # Calling method(args, kwargs) (line 24)
        method_call_result_236457 = invoke(stypy.reporting.localization.Localization(__file__, 24, 22), method_236446, *[function_236447, a_236448, b_236449], **kwargs_236456)
        
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___236458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), method_call_result_236457, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_236459 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), getitem___236458, int_236445)
        
        # Assigning a type to the variable 'tuple_var_assignment_236408' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'tuple_var_assignment_236408', subscript_call_result_236459)
        
        # Assigning a Subscript to a Name (line 24):
        
        # Obtaining the type of the subscript
        int_236460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        
        # Call to method(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'function' (line 24)
        function_236462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'function', False)
        # Getting the type of 'a' (line 24)
        a_236463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 39), 'a', False)
        # Getting the type of 'b' (line 24)
        b_236464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'b', False)
        # Processing the call keyword arguments (line 24)
        # Getting the type of 'xtol' (line 24)
        xtol_236465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 50), 'xtol', False)
        keyword_236466 = xtol_236465
        # Getting the type of 'rtol' (line 24)
        rtol_236467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 61), 'rtol', False)
        keyword_236468 = rtol_236467
        # Getting the type of 'True' (line 25)
        True_236469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 41), 'True', False)
        keyword_236470 = True_236469
        kwargs_236471 = {'xtol': keyword_236466, 'rtol': keyword_236468, 'full_output': keyword_236470}
        # Getting the type of 'method' (line 24)
        method_236461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'method', False)
        # Calling method(args, kwargs) (line 24)
        method_call_result_236472 = invoke(stypy.reporting.localization.Localization(__file__, 24, 22), method_236461, *[function_236462, a_236463, b_236464], **kwargs_236471)
        
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___236473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), method_call_result_236472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_236474 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), getitem___236473, int_236460)
        
        # Assigning a type to the variable 'tuple_var_assignment_236409' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'tuple_var_assignment_236409', subscript_call_result_236474)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_var_assignment_236408' (line 24)
        tuple_var_assignment_236408_236475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'tuple_var_assignment_236408')
        # Assigning a type to the variable 'zero' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'zero', tuple_var_assignment_236408_236475)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_var_assignment_236409' (line 24)
        tuple_var_assignment_236409_236476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'tuple_var_assignment_236409')
        # Assigning a type to the variable 'r' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'r', tuple_var_assignment_236409_236476)
        
        # Call to assert_(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'r' (line 26)
        r_236478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'r', False)
        # Obtaining the member 'converged' of a type (line 26)
        converged_236479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 20), r_236478, 'converged')
        # Processing the call keyword arguments (line 26)
        kwargs_236480 = {}
        # Getting the type of 'assert_' (line 26)
        assert__236477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 26)
        assert__call_result_236481 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), assert__236477, *[converged_236479], **kwargs_236480)
        
        
        # Call to assert_allclose(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'zero' (line 27)
        zero_236483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'zero', False)
        float_236484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'float')
        # Processing the call keyword arguments (line 27)
        # Getting the type of 'xtol' (line 27)
        xtol_236485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 44), 'xtol', False)
        keyword_236486 = xtol_236485
        # Getting the type of 'rtol' (line 27)
        rtol_236487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 55), 'rtol', False)
        keyword_236488 = rtol_236487
        str_236489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 24), 'str', 'method %s, function %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_236490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'name' (line 28)
        name_236491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 52), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 52), tuple_236490, name_236491)
        # Adding element type (line 28)
        # Getting the type of 'fname' (line 28)
        fname_236492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 58), 'fname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 52), tuple_236490, fname_236492)
        
        # Applying the binary operator '%' (line 28)
        result_mod_236493 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 24), '%', str_236489, tuple_236490)
        
        keyword_236494 = result_mod_236493
        kwargs_236495 = {'rtol': keyword_236488, 'err_msg': keyword_236494, 'atol': keyword_236486}
        # Getting the type of 'assert_allclose' (line 27)
        assert_allclose_236482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 27)
        assert_allclose_call_result_236496 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), assert_allclose_236482, *[zero_236483, float_236484], **kwargs_236495)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run_check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run_check' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_236497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run_check'
        return stypy_return_type_236497


    @norecursion
    def test_bisect(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bisect'
        module_type_store = module_type_store.open_function_context('test_bisect', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasic.test_bisect.__dict__.__setitem__('stypy_localization', localization)
        TestBasic.test_bisect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasic.test_bisect.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasic.test_bisect.__dict__.__setitem__('stypy_function_name', 'TestBasic.test_bisect')
        TestBasic.test_bisect.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasic.test_bisect.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasic.test_bisect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasic.test_bisect.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasic.test_bisect.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasic.test_bisect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasic.test_bisect.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasic.test_bisect', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bisect', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bisect(...)' code ##################

        
        # Call to run_check(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'cc' (line 31)
        cc_236500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'cc', False)
        # Obtaining the member 'bisect' of a type (line 31)
        bisect_236501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 23), cc_236500, 'bisect')
        str_236502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 34), 'str', 'bisect')
        # Processing the call keyword arguments (line 31)
        kwargs_236503 = {}
        # Getting the type of 'self' (line 31)
        self_236498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self', False)
        # Obtaining the member 'run_check' of a type (line 31)
        run_check_236499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_236498, 'run_check')
        # Calling run_check(args, kwargs) (line 31)
        run_check_call_result_236504 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), run_check_236499, *[bisect_236501, str_236502], **kwargs_236503)
        
        
        # ################# End of 'test_bisect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bisect' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_236505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bisect'
        return stypy_return_type_236505


    @norecursion
    def test_ridder(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ridder'
        module_type_store = module_type_store.open_function_context('test_ridder', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasic.test_ridder.__dict__.__setitem__('stypy_localization', localization)
        TestBasic.test_ridder.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasic.test_ridder.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasic.test_ridder.__dict__.__setitem__('stypy_function_name', 'TestBasic.test_ridder')
        TestBasic.test_ridder.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasic.test_ridder.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasic.test_ridder.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasic.test_ridder.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasic.test_ridder.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasic.test_ridder.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasic.test_ridder.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasic.test_ridder', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ridder', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ridder(...)' code ##################

        
        # Call to run_check(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'cc' (line 34)
        cc_236508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'cc', False)
        # Obtaining the member 'ridder' of a type (line 34)
        ridder_236509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 23), cc_236508, 'ridder')
        str_236510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'str', 'ridder')
        # Processing the call keyword arguments (line 34)
        kwargs_236511 = {}
        # Getting the type of 'self' (line 34)
        self_236506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'run_check' of a type (line 34)
        run_check_236507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_236506, 'run_check')
        # Calling run_check(args, kwargs) (line 34)
        run_check_call_result_236512 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), run_check_236507, *[ridder_236509, str_236510], **kwargs_236511)
        
        
        # ################# End of 'test_ridder(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ridder' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_236513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ridder'
        return stypy_return_type_236513


    @norecursion
    def test_brentq(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_brentq'
        module_type_store = module_type_store.open_function_context('test_brentq', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasic.test_brentq.__dict__.__setitem__('stypy_localization', localization)
        TestBasic.test_brentq.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasic.test_brentq.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasic.test_brentq.__dict__.__setitem__('stypy_function_name', 'TestBasic.test_brentq')
        TestBasic.test_brentq.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasic.test_brentq.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasic.test_brentq.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasic.test_brentq.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasic.test_brentq.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasic.test_brentq.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasic.test_brentq.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasic.test_brentq', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_brentq', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_brentq(...)' code ##################

        
        # Call to run_check(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'cc' (line 37)
        cc_236516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'cc', False)
        # Obtaining the member 'brentq' of a type (line 37)
        brentq_236517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 23), cc_236516, 'brentq')
        str_236518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'str', 'brentq')
        # Processing the call keyword arguments (line 37)
        kwargs_236519 = {}
        # Getting the type of 'self' (line 37)
        self_236514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self', False)
        # Obtaining the member 'run_check' of a type (line 37)
        run_check_236515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_236514, 'run_check')
        # Calling run_check(args, kwargs) (line 37)
        run_check_call_result_236520 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), run_check_236515, *[brentq_236517, str_236518], **kwargs_236519)
        
        
        # ################# End of 'test_brentq(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_brentq' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_236521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236521)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_brentq'
        return stypy_return_type_236521


    @norecursion
    def test_brenth(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_brenth'
        module_type_store = module_type_store.open_function_context('test_brenth', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasic.test_brenth.__dict__.__setitem__('stypy_localization', localization)
        TestBasic.test_brenth.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasic.test_brenth.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasic.test_brenth.__dict__.__setitem__('stypy_function_name', 'TestBasic.test_brenth')
        TestBasic.test_brenth.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasic.test_brenth.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasic.test_brenth.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasic.test_brenth.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasic.test_brenth.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasic.test_brenth.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasic.test_brenth.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasic.test_brenth', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_brenth', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_brenth(...)' code ##################

        
        # Call to run_check(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'cc' (line 40)
        cc_236524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'cc', False)
        # Obtaining the member 'brenth' of a type (line 40)
        brenth_236525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 23), cc_236524, 'brenth')
        str_236526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'str', 'brenth')
        # Processing the call keyword arguments (line 40)
        kwargs_236527 = {}
        # Getting the type of 'self' (line 40)
        self_236522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member 'run_check' of a type (line 40)
        run_check_236523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_236522, 'run_check')
        # Calling run_check(args, kwargs) (line 40)
        run_check_call_result_236528 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), run_check_236523, *[brenth_236525, str_236526], **kwargs_236527)
        
        
        # ################# End of 'test_brenth(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_brenth' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_236529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236529)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_brenth'
        return stypy_return_type_236529


    @norecursion
    def test_newton(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_newton'
        module_type_store = module_type_store.open_function_context('test_newton', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasic.test_newton.__dict__.__setitem__('stypy_localization', localization)
        TestBasic.test_newton.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasic.test_newton.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasic.test_newton.__dict__.__setitem__('stypy_function_name', 'TestBasic.test_newton')
        TestBasic.test_newton.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasic.test_newton.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasic.test_newton.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasic.test_newton.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasic.test_newton.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasic.test_newton.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasic.test_newton.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasic.test_newton', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_newton', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_newton(...)' code ##################

        
        # Assigning a Lambda to a Name (line 43):
        
        # Assigning a Lambda to a Name (line 43):

        @norecursion
        def _stypy_temp_lambda_157(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_157'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_157', 43, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_157.stypy_localization = localization
            _stypy_temp_lambda_157.stypy_type_of_self = None
            _stypy_temp_lambda_157.stypy_type_store = module_type_store
            _stypy_temp_lambda_157.stypy_function_name = '_stypy_temp_lambda_157'
            _stypy_temp_lambda_157.stypy_param_names_list = ['x']
            _stypy_temp_lambda_157.stypy_varargs_param_name = None
            _stypy_temp_lambda_157.stypy_kwargs_param_name = None
            _stypy_temp_lambda_157.stypy_call_defaults = defaults
            _stypy_temp_lambda_157.stypy_call_varargs = varargs
            _stypy_temp_lambda_157.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_157', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_157', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 43)
            x_236530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'x')
            int_236531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'int')
            # Applying the binary operator '**' (line 43)
            result_pow_236532 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 23), '**', x_236530, int_236531)
            
            int_236533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'int')
            # Getting the type of 'x' (line 43)
            x_236534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'x')
            # Applying the binary operator '*' (line 43)
            result_mul_236535 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 30), '*', int_236533, x_236534)
            
            # Applying the binary operator '-' (line 43)
            result_sub_236536 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 23), '-', result_pow_236532, result_mul_236535)
            
            int_236537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'int')
            # Applying the binary operator '-' (line 43)
            result_sub_236538 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 34), '-', result_sub_236536, int_236537)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'stypy_return_type', result_sub_236538)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_157' in the type store
            # Getting the type of 'stypy_return_type' (line 43)
            stypy_return_type_236539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236539)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_157'
            return stypy_return_type_236539

        # Assigning a type to the variable '_stypy_temp_lambda_157' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), '_stypy_temp_lambda_157', _stypy_temp_lambda_157)
        # Getting the type of '_stypy_temp_lambda_157' (line 43)
        _stypy_temp_lambda_157_236540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), '_stypy_temp_lambda_157')
        # Assigning a type to the variable 'f1' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'f1', _stypy_temp_lambda_157_236540)
        
        # Assigning a Lambda to a Name (line 44):
        
        # Assigning a Lambda to a Name (line 44):

        @norecursion
        def _stypy_temp_lambda_158(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_158'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_158', 44, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_158.stypy_localization = localization
            _stypy_temp_lambda_158.stypy_type_of_self = None
            _stypy_temp_lambda_158.stypy_type_store = module_type_store
            _stypy_temp_lambda_158.stypy_function_name = '_stypy_temp_lambda_158'
            _stypy_temp_lambda_158.stypy_param_names_list = ['x']
            _stypy_temp_lambda_158.stypy_varargs_param_name = None
            _stypy_temp_lambda_158.stypy_kwargs_param_name = None
            _stypy_temp_lambda_158.stypy_call_defaults = defaults
            _stypy_temp_lambda_158.stypy_call_varargs = varargs
            _stypy_temp_lambda_158.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_158', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_158', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_236541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'int')
            # Getting the type of 'x' (line 44)
            x_236542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'x')
            # Applying the binary operator '*' (line 44)
            result_mul_236543 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 25), '*', int_236541, x_236542)
            
            int_236544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'int')
            # Applying the binary operator '-' (line 44)
            result_sub_236545 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 25), '-', result_mul_236543, int_236544)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'stypy_return_type', result_sub_236545)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_158' in the type store
            # Getting the type of 'stypy_return_type' (line 44)
            stypy_return_type_236546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236546)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_158'
            return stypy_return_type_236546

        # Assigning a type to the variable '_stypy_temp_lambda_158' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), '_stypy_temp_lambda_158', _stypy_temp_lambda_158)
        # Getting the type of '_stypy_temp_lambda_158' (line 44)
        _stypy_temp_lambda_158_236547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), '_stypy_temp_lambda_158')
        # Assigning a type to the variable 'f1_1' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'f1_1', _stypy_temp_lambda_158_236547)
        
        # Assigning a Lambda to a Name (line 45):
        
        # Assigning a Lambda to a Name (line 45):

        @norecursion
        def _stypy_temp_lambda_159(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_159'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_159', 45, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_159.stypy_localization = localization
            _stypy_temp_lambda_159.stypy_type_of_self = None
            _stypy_temp_lambda_159.stypy_type_store = module_type_store
            _stypy_temp_lambda_159.stypy_function_name = '_stypy_temp_lambda_159'
            _stypy_temp_lambda_159.stypy_param_names_list = ['x']
            _stypy_temp_lambda_159.stypy_varargs_param_name = None
            _stypy_temp_lambda_159.stypy_kwargs_param_name = None
            _stypy_temp_lambda_159.stypy_call_defaults = defaults
            _stypy_temp_lambda_159.stypy_call_varargs = varargs
            _stypy_temp_lambda_159.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_159', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_159', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            float_236548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'float')
            int_236549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'int')
            # Getting the type of 'x' (line 45)
            x_236550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'x')
            # Applying the binary operator '*' (line 45)
            result_mul_236551 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 31), '*', int_236549, x_236550)
            
            # Applying the binary operator '+' (line 45)
            result_add_236552 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 25), '+', float_236548, result_mul_236551)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'stypy_return_type', result_add_236552)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_159' in the type store
            # Getting the type of 'stypy_return_type' (line 45)
            stypy_return_type_236553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236553)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_159'
            return stypy_return_type_236553

        # Assigning a type to the variable '_stypy_temp_lambda_159' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), '_stypy_temp_lambda_159', _stypy_temp_lambda_159)
        # Getting the type of '_stypy_temp_lambda_159' (line 45)
        _stypy_temp_lambda_159_236554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), '_stypy_temp_lambda_159')
        # Assigning a type to the variable 'f1_2' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'f1_2', _stypy_temp_lambda_159_236554)
        
        # Assigning a Lambda to a Name (line 47):
        
        # Assigning a Lambda to a Name (line 47):

        @norecursion
        def _stypy_temp_lambda_160(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_160'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_160', 47, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_160.stypy_localization = localization
            _stypy_temp_lambda_160.stypy_type_of_self = None
            _stypy_temp_lambda_160.stypy_type_store = module_type_store
            _stypy_temp_lambda_160.stypy_function_name = '_stypy_temp_lambda_160'
            _stypy_temp_lambda_160.stypy_param_names_list = ['x']
            _stypy_temp_lambda_160.stypy_varargs_param_name = None
            _stypy_temp_lambda_160.stypy_kwargs_param_name = None
            _stypy_temp_lambda_160.stypy_call_defaults = defaults
            _stypy_temp_lambda_160.stypy_call_varargs = varargs
            _stypy_temp_lambda_160.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_160', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_160', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to exp(...): (line 47)
            # Processing the call arguments (line 47)
            # Getting the type of 'x' (line 47)
            x_236556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'x', False)
            # Processing the call keyword arguments (line 47)
            kwargs_236557 = {}
            # Getting the type of 'exp' (line 47)
            exp_236555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'exp', False)
            # Calling exp(args, kwargs) (line 47)
            exp_call_result_236558 = invoke(stypy.reporting.localization.Localization(__file__, 47, 23), exp_236555, *[x_236556], **kwargs_236557)
            
            
            # Call to cos(...): (line 47)
            # Processing the call arguments (line 47)
            # Getting the type of 'x' (line 47)
            x_236560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 36), 'x', False)
            # Processing the call keyword arguments (line 47)
            kwargs_236561 = {}
            # Getting the type of 'cos' (line 47)
            cos_236559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 32), 'cos', False)
            # Calling cos(args, kwargs) (line 47)
            cos_call_result_236562 = invoke(stypy.reporting.localization.Localization(__file__, 47, 32), cos_236559, *[x_236560], **kwargs_236561)
            
            # Applying the binary operator '-' (line 47)
            result_sub_236563 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 23), '-', exp_call_result_236558, cos_call_result_236562)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'stypy_return_type', result_sub_236563)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_160' in the type store
            # Getting the type of 'stypy_return_type' (line 47)
            stypy_return_type_236564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236564)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_160'
            return stypy_return_type_236564

        # Assigning a type to the variable '_stypy_temp_lambda_160' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), '_stypy_temp_lambda_160', _stypy_temp_lambda_160)
        # Getting the type of '_stypy_temp_lambda_160' (line 47)
        _stypy_temp_lambda_160_236565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), '_stypy_temp_lambda_160')
        # Assigning a type to the variable 'f2' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'f2', _stypy_temp_lambda_160_236565)
        
        # Assigning a Lambda to a Name (line 48):
        
        # Assigning a Lambda to a Name (line 48):

        @norecursion
        def _stypy_temp_lambda_161(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_161'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_161', 48, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_161.stypy_localization = localization
            _stypy_temp_lambda_161.stypy_type_of_self = None
            _stypy_temp_lambda_161.stypy_type_store = module_type_store
            _stypy_temp_lambda_161.stypy_function_name = '_stypy_temp_lambda_161'
            _stypy_temp_lambda_161.stypy_param_names_list = ['x']
            _stypy_temp_lambda_161.stypy_varargs_param_name = None
            _stypy_temp_lambda_161.stypy_kwargs_param_name = None
            _stypy_temp_lambda_161.stypy_call_defaults = defaults
            _stypy_temp_lambda_161.stypy_call_varargs = varargs
            _stypy_temp_lambda_161.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_161', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_161', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to exp(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'x' (line 48)
            x_236567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'x', False)
            # Processing the call keyword arguments (line 48)
            kwargs_236568 = {}
            # Getting the type of 'exp' (line 48)
            exp_236566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'exp', False)
            # Calling exp(args, kwargs) (line 48)
            exp_call_result_236569 = invoke(stypy.reporting.localization.Localization(__file__, 48, 25), exp_236566, *[x_236567], **kwargs_236568)
            
            
            # Call to sin(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'x' (line 48)
            x_236571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 38), 'x', False)
            # Processing the call keyword arguments (line 48)
            kwargs_236572 = {}
            # Getting the type of 'sin' (line 48)
            sin_236570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'sin', False)
            # Calling sin(args, kwargs) (line 48)
            sin_call_result_236573 = invoke(stypy.reporting.localization.Localization(__file__, 48, 34), sin_236570, *[x_236571], **kwargs_236572)
            
            # Applying the binary operator '+' (line 48)
            result_add_236574 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 25), '+', exp_call_result_236569, sin_call_result_236573)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'stypy_return_type', result_add_236574)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_161' in the type store
            # Getting the type of 'stypy_return_type' (line 48)
            stypy_return_type_236575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236575)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_161'
            return stypy_return_type_236575

        # Assigning a type to the variable '_stypy_temp_lambda_161' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), '_stypy_temp_lambda_161', _stypy_temp_lambda_161)
        # Getting the type of '_stypy_temp_lambda_161' (line 48)
        _stypy_temp_lambda_161_236576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), '_stypy_temp_lambda_161')
        # Assigning a type to the variable 'f2_1' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'f2_1', _stypy_temp_lambda_161_236576)
        
        # Assigning a Lambda to a Name (line 49):
        
        # Assigning a Lambda to a Name (line 49):

        @norecursion
        def _stypy_temp_lambda_162(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_162'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_162', 49, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_162.stypy_localization = localization
            _stypy_temp_lambda_162.stypy_type_of_self = None
            _stypy_temp_lambda_162.stypy_type_store = module_type_store
            _stypy_temp_lambda_162.stypy_function_name = '_stypy_temp_lambda_162'
            _stypy_temp_lambda_162.stypy_param_names_list = ['x']
            _stypy_temp_lambda_162.stypy_varargs_param_name = None
            _stypy_temp_lambda_162.stypy_kwargs_param_name = None
            _stypy_temp_lambda_162.stypy_call_defaults = defaults
            _stypy_temp_lambda_162.stypy_call_varargs = varargs
            _stypy_temp_lambda_162.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_162', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_162', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to exp(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'x' (line 49)
            x_236578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'x', False)
            # Processing the call keyword arguments (line 49)
            kwargs_236579 = {}
            # Getting the type of 'exp' (line 49)
            exp_236577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'exp', False)
            # Calling exp(args, kwargs) (line 49)
            exp_call_result_236580 = invoke(stypy.reporting.localization.Localization(__file__, 49, 25), exp_236577, *[x_236578], **kwargs_236579)
            
            
            # Call to cos(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'x' (line 49)
            x_236582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x', False)
            # Processing the call keyword arguments (line 49)
            kwargs_236583 = {}
            # Getting the type of 'cos' (line 49)
            cos_236581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'cos', False)
            # Calling cos(args, kwargs) (line 49)
            cos_call_result_236584 = invoke(stypy.reporting.localization.Localization(__file__, 49, 34), cos_236581, *[x_236582], **kwargs_236583)
            
            # Applying the binary operator '+' (line 49)
            result_add_236585 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 25), '+', exp_call_result_236580, cos_call_result_236584)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'stypy_return_type', result_add_236585)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_162' in the type store
            # Getting the type of 'stypy_return_type' (line 49)
            stypy_return_type_236586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236586)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_162'
            return stypy_return_type_236586

        # Assigning a type to the variable '_stypy_temp_lambda_162' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), '_stypy_temp_lambda_162', _stypy_temp_lambda_162)
        # Getting the type of '_stypy_temp_lambda_162' (line 49)
        _stypy_temp_lambda_162_236587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), '_stypy_temp_lambda_162')
        # Assigning a type to the variable 'f2_2' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'f2_2', _stypy_temp_lambda_162_236587)
        
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_236588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_236589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        # Adding element type (line 51)
        # Getting the type of 'f1' (line 51)
        f1_236590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'f1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 29), tuple_236589, f1_236590)
        # Adding element type (line 51)
        # Getting the type of 'f1_1' (line 51)
        f1_1_236591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'f1_1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 29), tuple_236589, f1_1_236591)
        # Adding element type (line 51)
        # Getting the type of 'f1_2' (line 51)
        f1_2_236592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 39), 'f1_2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 29), tuple_236589, f1_2_236592)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_236588, tuple_236589)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_236593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        # Adding element type (line 51)
        # Getting the type of 'f2' (line 51)
        f2_236594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 47), 'f2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 47), tuple_236593, f2_236594)
        # Adding element type (line 51)
        # Getting the type of 'f2_1' (line 51)
        f2_1_236595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 51), 'f2_1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 47), tuple_236593, f2_1_236595)
        # Adding element type (line 51)
        # Getting the type of 'f2_2' (line 51)
        f2_2_236596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 57), 'f2_2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 47), tuple_236593, f2_2_236596)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_236588, tuple_236593)
        
        # Testing the type of a for loop iterable (line 51)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 8), list_236588)
        # Getting the type of the for loop variable (line 51)
        for_loop_var_236597 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 8), list_236588)
        # Assigning a type to the variable 'f' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), for_loop_var_236597))
        # Assigning a type to the variable 'f_1' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'f_1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), for_loop_var_236597))
        # Assigning a type to the variable 'f_2' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'f_2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), for_loop_var_236597))
        # SSA begins for a for statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to newton(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'f' (line 52)
        f_236600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'f', False)
        int_236601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'int')
        # Processing the call keyword arguments (line 52)
        float_236602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 39), 'float')
        keyword_236603 = float_236602
        kwargs_236604 = {'tol': keyword_236603}
        # Getting the type of 'zeros' (line 52)
        zeros_236598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'zeros', False)
        # Obtaining the member 'newton' of a type (line 52)
        newton_236599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), zeros_236598, 'newton')
        # Calling newton(args, kwargs) (line 52)
        newton_call_result_236605 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), newton_236599, *[f_236600, int_236601], **kwargs_236604)
        
        # Assigning a type to the variable 'x' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'x', newton_call_result_236605)
        
        # Call to assert_allclose(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to f(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'x' (line 53)
        x_236608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'x', False)
        # Processing the call keyword arguments (line 53)
        kwargs_236609 = {}
        # Getting the type of 'f' (line 53)
        f_236607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'f', False)
        # Calling f(args, kwargs) (line 53)
        f_call_result_236610 = invoke(stypy.reporting.localization.Localization(__file__, 53, 28), f_236607, *[x_236608], **kwargs_236609)
        
        int_236611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'int')
        # Processing the call keyword arguments (line 53)
        float_236612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 42), 'float')
        keyword_236613 = float_236612
        kwargs_236614 = {'atol': keyword_236613}
        # Getting the type of 'assert_allclose' (line 53)
        assert_allclose_236606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 53)
        assert_allclose_call_result_236615 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), assert_allclose_236606, *[f_call_result_236610, int_236611], **kwargs_236614)
        
        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to newton(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'f' (line 54)
        f_236618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'f', False)
        int_236619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 32), 'int')
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'f_1' (line 54)
        f_1_236620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'f_1', False)
        keyword_236621 = f_1_236620
        float_236622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 51), 'float')
        keyword_236623 = float_236622
        kwargs_236624 = {'fprime': keyword_236621, 'tol': keyword_236623}
        # Getting the type of 'zeros' (line 54)
        zeros_236616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'zeros', False)
        # Obtaining the member 'newton' of a type (line 54)
        newton_236617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), zeros_236616, 'newton')
        # Calling newton(args, kwargs) (line 54)
        newton_call_result_236625 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), newton_236617, *[f_236618, int_236619], **kwargs_236624)
        
        # Assigning a type to the variable 'x' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'x', newton_call_result_236625)
        
        # Call to assert_allclose(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Call to f(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'x' (line 55)
        x_236628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 30), 'x', False)
        # Processing the call keyword arguments (line 55)
        kwargs_236629 = {}
        # Getting the type of 'f' (line 55)
        f_236627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'f', False)
        # Calling f(args, kwargs) (line 55)
        f_call_result_236630 = invoke(stypy.reporting.localization.Localization(__file__, 55, 28), f_236627, *[x_236628], **kwargs_236629)
        
        int_236631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
        # Processing the call keyword arguments (line 55)
        float_236632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'float')
        keyword_236633 = float_236632
        kwargs_236634 = {'atol': keyword_236633}
        # Getting the type of 'assert_allclose' (line 55)
        assert_allclose_236626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 55)
        assert_allclose_call_result_236635 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), assert_allclose_236626, *[f_call_result_236630, int_236631], **kwargs_236634)
        
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to newton(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'f' (line 56)
        f_236638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'f', False)
        int_236639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 32), 'int')
        # Processing the call keyword arguments (line 56)
        # Getting the type of 'f_1' (line 56)
        f_1_236640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 42), 'f_1', False)
        keyword_236641 = f_1_236640
        # Getting the type of 'f_2' (line 56)
        f_2_236642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 55), 'f_2', False)
        keyword_236643 = f_2_236642
        float_236644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 64), 'float')
        keyword_236645 = float_236644
        kwargs_236646 = {'fprime2': keyword_236643, 'fprime': keyword_236641, 'tol': keyword_236645}
        # Getting the type of 'zeros' (line 56)
        zeros_236636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'zeros', False)
        # Obtaining the member 'newton' of a type (line 56)
        newton_236637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), zeros_236636, 'newton')
        # Calling newton(args, kwargs) (line 56)
        newton_call_result_236647 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), newton_236637, *[f_236638, int_236639], **kwargs_236646)
        
        # Assigning a type to the variable 'x' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'x', newton_call_result_236647)
        
        # Call to assert_allclose(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to f(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'x' (line 57)
        x_236650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'x', False)
        # Processing the call keyword arguments (line 57)
        kwargs_236651 = {}
        # Getting the type of 'f' (line 57)
        f_236649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'f', False)
        # Calling f(args, kwargs) (line 57)
        f_call_result_236652 = invoke(stypy.reporting.localization.Localization(__file__, 57, 28), f_236649, *[x_236650], **kwargs_236651)
        
        int_236653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'int')
        # Processing the call keyword arguments (line 57)
        float_236654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 42), 'float')
        keyword_236655 = float_236654
        kwargs_236656 = {'atol': keyword_236655}
        # Getting the type of 'assert_allclose' (line 57)
        assert_allclose_236648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 57)
        assert_allclose_call_result_236657 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), assert_allclose_236648, *[f_call_result_236652, int_236653], **kwargs_236656)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_newton(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_newton' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_236658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236658)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_newton'
        return stypy_return_type_236658


    @norecursion
    def test_deriv_zero_warning(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_deriv_zero_warning'
        module_type_store = module_type_store.open_function_context('test_deriv_zero_warning', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_localization', localization)
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_function_name', 'TestBasic.test_deriv_zero_warning')
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasic.test_deriv_zero_warning.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasic.test_deriv_zero_warning', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_deriv_zero_warning', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_deriv_zero_warning(...)' code ##################

        
        # Assigning a Lambda to a Name (line 60):
        
        # Assigning a Lambda to a Name (line 60):

        @norecursion
        def _stypy_temp_lambda_163(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_163'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_163', 60, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_163.stypy_localization = localization
            _stypy_temp_lambda_163.stypy_type_of_self = None
            _stypy_temp_lambda_163.stypy_type_store = module_type_store
            _stypy_temp_lambda_163.stypy_function_name = '_stypy_temp_lambda_163'
            _stypy_temp_lambda_163.stypy_param_names_list = ['x']
            _stypy_temp_lambda_163.stypy_varargs_param_name = None
            _stypy_temp_lambda_163.stypy_kwargs_param_name = None
            _stypy_temp_lambda_163.stypy_call_defaults = defaults
            _stypy_temp_lambda_163.stypy_call_varargs = varargs
            _stypy_temp_lambda_163.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_163', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_163', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 60)
            x_236659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'x')
            int_236660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
            # Applying the binary operator '**' (line 60)
            result_pow_236661 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 25), '**', x_236659, int_236660)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'stypy_return_type', result_pow_236661)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_163' in the type store
            # Getting the type of 'stypy_return_type' (line 60)
            stypy_return_type_236662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236662)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_163'
            return stypy_return_type_236662

        # Assigning a type to the variable '_stypy_temp_lambda_163' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), '_stypy_temp_lambda_163', _stypy_temp_lambda_163)
        # Getting the type of '_stypy_temp_lambda_163' (line 60)
        _stypy_temp_lambda_163_236663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), '_stypy_temp_lambda_163')
        # Assigning a type to the variable 'func' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'func', _stypy_temp_lambda_163_236663)
        
        # Assigning a Lambda to a Name (line 61):
        
        # Assigning a Lambda to a Name (line 61):

        @norecursion
        def _stypy_temp_lambda_164(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_164'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_164', 61, 16, True)
            # Passed parameters checking function
            _stypy_temp_lambda_164.stypy_localization = localization
            _stypy_temp_lambda_164.stypy_type_of_self = None
            _stypy_temp_lambda_164.stypy_type_store = module_type_store
            _stypy_temp_lambda_164.stypy_function_name = '_stypy_temp_lambda_164'
            _stypy_temp_lambda_164.stypy_param_names_list = ['x']
            _stypy_temp_lambda_164.stypy_varargs_param_name = None
            _stypy_temp_lambda_164.stypy_kwargs_param_name = None
            _stypy_temp_lambda_164.stypy_call_defaults = defaults
            _stypy_temp_lambda_164.stypy_call_varargs = varargs
            _stypy_temp_lambda_164.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_164', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_164', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_236664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'int')
            # Getting the type of 'x' (line 61)
            x_236665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 28), 'x')
            # Applying the binary operator '*' (line 61)
            result_mul_236666 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 26), '*', int_236664, x_236665)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'stypy_return_type', result_mul_236666)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_164' in the type store
            # Getting the type of 'stypy_return_type' (line 61)
            stypy_return_type_236667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236667)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_164'
            return stypy_return_type_236667

        # Assigning a type to the variable '_stypy_temp_lambda_164' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), '_stypy_temp_lambda_164', _stypy_temp_lambda_164)
        # Getting the type of '_stypy_temp_lambda_164' (line 61)
        _stypy_temp_lambda_164_236668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), '_stypy_temp_lambda_164')
        # Assigning a type to the variable 'dfunc' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'dfunc', _stypy_temp_lambda_164_236668)
        
        # Call to assert_warns(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'RuntimeWarning' (line 62)
        RuntimeWarning_236670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'RuntimeWarning', False)
        # Getting the type of 'cc' (line 62)
        cc_236671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 37), 'cc', False)
        # Obtaining the member 'newton' of a type (line 62)
        newton_236672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 37), cc_236671, 'newton')
        # Getting the type of 'func' (line 62)
        func_236673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 48), 'func', False)
        float_236674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 54), 'float')
        # Getting the type of 'dfunc' (line 62)
        dfunc_236675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 59), 'dfunc', False)
        # Processing the call keyword arguments (line 62)
        kwargs_236676 = {}
        # Getting the type of 'assert_warns' (line 62)
        assert_warns_236669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 62)
        assert_warns_call_result_236677 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_warns_236669, *[RuntimeWarning_236670, newton_236672, func_236673, float_236674, dfunc_236675], **kwargs_236676)
        
        
        # ################# End of 'test_deriv_zero_warning(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_deriv_zero_warning' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_236678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_deriv_zero_warning'
        return stypy_return_type_236678


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 0, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasic.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBasic' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'TestBasic', TestBasic)

@norecursion
def test_gh_5555(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gh_5555'
    module_type_store = module_type_store.open_function_context('test_gh_5555', 65, 0, False)
    
    # Passed parameters checking function
    test_gh_5555.stypy_localization = localization
    test_gh_5555.stypy_type_of_self = None
    test_gh_5555.stypy_type_store = module_type_store
    test_gh_5555.stypy_function_name = 'test_gh_5555'
    test_gh_5555.stypy_param_names_list = []
    test_gh_5555.stypy_varargs_param_name = None
    test_gh_5555.stypy_kwargs_param_name = None
    test_gh_5555.stypy_call_defaults = defaults
    test_gh_5555.stypy_call_varargs = varargs
    test_gh_5555.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gh_5555', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gh_5555', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gh_5555(...)' code ##################

    
    # Assigning a Num to a Name (line 66):
    
    # Assigning a Num to a Name (line 66):
    float_236679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 11), 'float')
    # Assigning a type to the variable 'root' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'root', float_236679)

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 68, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['x']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        # Getting the type of 'x' (line 69)
        x_236680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'x')
        # Getting the type of 'root' (line 69)
        root_236681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'root')
        # Applying the binary operator '-' (line 69)
        result_sub_236682 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 15), '-', x_236680, root_236681)
        
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', result_sub_236682)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_236683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236683)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_236683

    # Assigning a type to the variable 'f' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'f', f)
    
    # Assigning a List to a Name (line 71):
    
    # Assigning a List to a Name (line 71):
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_236684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'cc' (line 71)
    cc_236685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'cc')
    # Obtaining the member 'bisect' of a type (line 71)
    bisect_236686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 15), cc_236685, 'bisect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 14), list_236684, bisect_236686)
    # Adding element type (line 71)
    # Getting the type of 'cc' (line 71)
    cc_236687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'cc')
    # Obtaining the member 'ridder' of a type (line 71)
    ridder_236688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), cc_236687, 'ridder')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 14), list_236684, ridder_236688)
    
    # Assigning a type to the variable 'methods' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'methods', list_236684)
    
    # Assigning a BinOp to a Name (line 72):
    
    # Assigning a BinOp to a Name (line 72):
    int_236689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'int')
    
    # Call to finfo(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'float' (line 72)
    float_236691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'float', False)
    # Processing the call keyword arguments (line 72)
    kwargs_236692 = {}
    # Getting the type of 'finfo' (line 72)
    finfo_236690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'finfo', False)
    # Calling finfo(args, kwargs) (line 72)
    finfo_call_result_236693 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), finfo_236690, *[float_236691], **kwargs_236692)
    
    # Obtaining the member 'eps' of a type (line 72)
    eps_236694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), finfo_call_result_236693, 'eps')
    # Applying the binary operator '*' (line 72)
    result_mul_236695 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), '*', int_236689, eps_236694)
    
    # Assigning a type to the variable 'xtol' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'xtol', result_mul_236695)
    
    # Assigning a BinOp to a Name (line 73):
    
    # Assigning a BinOp to a Name (line 73):
    int_236696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 11), 'int')
    
    # Call to finfo(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'float' (line 73)
    float_236698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'float', False)
    # Processing the call keyword arguments (line 73)
    kwargs_236699 = {}
    # Getting the type of 'finfo' (line 73)
    finfo_236697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'finfo', False)
    # Calling finfo(args, kwargs) (line 73)
    finfo_call_result_236700 = invoke(stypy.reporting.localization.Localization(__file__, 73, 13), finfo_236697, *[float_236698], **kwargs_236699)
    
    # Obtaining the member 'eps' of a type (line 73)
    eps_236701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 13), finfo_call_result_236700, 'eps')
    # Applying the binary operator '*' (line 73)
    result_mul_236702 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), '*', int_236696, eps_236701)
    
    # Assigning a type to the variable 'rtol' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'rtol', result_mul_236702)
    
    # Getting the type of 'methods' (line 74)
    methods_236703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'methods')
    # Testing the type of a for loop iterable (line 74)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 4), methods_236703)
    # Getting the type of the for loop variable (line 74)
    for_loop_var_236704 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 4), methods_236703)
    # Assigning a type to the variable 'method' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'method', for_loop_var_236704)
    # SSA begins for a for statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to method(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'f' (line 75)
    f_236706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'f', False)
    float_236707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'float')
    float_236708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 30), 'float')
    # Processing the call keyword arguments (line 75)
    # Getting the type of 'xtol' (line 75)
    xtol_236709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'xtol', False)
    keyword_236710 = xtol_236709
    # Getting the type of 'rtol' (line 75)
    rtol_236711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 51), 'rtol', False)
    keyword_236712 = rtol_236711
    kwargs_236713 = {'xtol': keyword_236710, 'rtol': keyword_236712}
    # Getting the type of 'method' (line 75)
    method_236705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'method', False)
    # Calling method(args, kwargs) (line 75)
    method_call_result_236714 = invoke(stypy.reporting.localization.Localization(__file__, 75, 14), method_236705, *[f_236706, float_236707, float_236708], **kwargs_236713)
    
    # Assigning a type to the variable 'res' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'res', method_call_result_236714)
    
    # Call to assert_allclose(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'root' (line 76)
    root_236716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'root', False)
    # Getting the type of 'res' (line 76)
    res_236717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'res', False)
    # Processing the call keyword arguments (line 76)
    # Getting the type of 'xtol' (line 76)
    xtol_236718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 40), 'xtol', False)
    keyword_236719 = xtol_236718
    # Getting the type of 'rtol' (line 76)
    rtol_236720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 51), 'rtol', False)
    keyword_236721 = rtol_236720
    str_236722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'str', 'method %s')
    # Getting the type of 'method' (line 77)
    method_236723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 46), 'method', False)
    # Obtaining the member '__name__' of a type (line 77)
    name___236724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 46), method_236723, '__name__')
    # Applying the binary operator '%' (line 77)
    result_mod_236725 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 32), '%', str_236722, name___236724)
    
    keyword_236726 = result_mod_236725
    kwargs_236727 = {'rtol': keyword_236721, 'err_msg': keyword_236726, 'atol': keyword_236719}
    # Getting the type of 'assert_allclose' (line 76)
    assert_allclose_236715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 76)
    assert_allclose_call_result_236728 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_allclose_236715, *[root_236716, res_236717], **kwargs_236727)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_gh_5555(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gh_5555' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_236729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_236729)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gh_5555'
    return stypy_return_type_236729

# Assigning a type to the variable 'test_gh_5555' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'test_gh_5555', test_gh_5555)

@norecursion
def test_gh_5557(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gh_5557'
    module_type_store = module_type_store.open_function_context('test_gh_5557', 80, 0, False)
    
    # Passed parameters checking function
    test_gh_5557.stypy_localization = localization
    test_gh_5557.stypy_type_of_self = None
    test_gh_5557.stypy_type_store = module_type_store
    test_gh_5557.stypy_function_name = 'test_gh_5557'
    test_gh_5557.stypy_param_names_list = []
    test_gh_5557.stypy_varargs_param_name = None
    test_gh_5557.stypy_kwargs_param_name = None
    test_gh_5557.stypy_call_defaults = defaults
    test_gh_5557.stypy_call_varargs = varargs
    test_gh_5557.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gh_5557', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gh_5557', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gh_5557(...)' code ##################


    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 90, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['x']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        
        # Getting the type of 'x' (line 91)
        x_236730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'x')
        float_236731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'float')
        # Applying the binary operator '<' (line 91)
        result_lt_236732 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), '<', x_236730, float_236731)
        
        # Testing the type of an if condition (line 91)
        if_condition_236733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), result_lt_236732)
        # Assigning a type to the variable 'if_condition_236733' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_236733', if_condition_236733)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_236734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'stypy_return_type', float_236734)
        # SSA branch for the else part of an if statement (line 91)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'x' (line 94)
        x_236735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'x')
        float_236736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'float')
        # Applying the binary operator '-' (line 94)
        result_sub_236737 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 19), '-', x_236735, float_236736)
        
        # Assigning a type to the variable 'stypy_return_type' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'stypy_return_type', result_sub_236737)
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_236738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_236738

    # Assigning a type to the variable 'f' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'f', f)
    
    # Assigning a Num to a Name (line 96):
    
    # Assigning a Num to a Name (line 96):
    float_236739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'float')
    # Assigning a type to the variable 'atol' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'atol', float_236739)
    
    # Assigning a BinOp to a Name (line 97):
    
    # Assigning a BinOp to a Name (line 97):
    int_236740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 11), 'int')
    
    # Call to finfo(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'float' (line 97)
    float_236742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'float', False)
    # Processing the call keyword arguments (line 97)
    kwargs_236743 = {}
    # Getting the type of 'finfo' (line 97)
    finfo_236741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'finfo', False)
    # Calling finfo(args, kwargs) (line 97)
    finfo_call_result_236744 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), finfo_236741, *[float_236742], **kwargs_236743)
    
    # Obtaining the member 'eps' of a type (line 97)
    eps_236745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 13), finfo_call_result_236744, 'eps')
    # Applying the binary operator '*' (line 97)
    result_mul_236746 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), '*', int_236740, eps_236745)
    
    # Assigning a type to the variable 'rtol' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'rtol', result_mul_236746)
    
    # Assigning a List to a Name (line 98):
    
    # Assigning a List to a Name (line 98):
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_236747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    # Getting the type of 'cc' (line 98)
    cc_236748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'cc')
    # Obtaining the member 'brentq' of a type (line 98)
    brentq_236749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), cc_236748, 'brentq')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 14), list_236747, brentq_236749)
    # Adding element type (line 98)
    # Getting the type of 'cc' (line 98)
    cc_236750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'cc')
    # Obtaining the member 'brenth' of a type (line 98)
    brenth_236751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 26), cc_236750, 'brenth')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 14), list_236747, brenth_236751)
    
    # Assigning a type to the variable 'methods' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'methods', list_236747)
    
    # Getting the type of 'methods' (line 99)
    methods_236752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'methods')
    # Testing the type of a for loop iterable (line 99)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 4), methods_236752)
    # Getting the type of the for loop variable (line 99)
    for_loop_var_236753 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 4), methods_236752)
    # Assigning a type to the variable 'method' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'method', for_loop_var_236753)
    # SSA begins for a for statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to method(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'f' (line 100)
    f_236755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'f', False)
    int_236756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'int')
    int_236757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'int')
    # Processing the call keyword arguments (line 100)
    # Getting the type of 'atol' (line 100)
    atol_236758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 'atol', False)
    keyword_236759 = atol_236758
    # Getting the type of 'rtol' (line 100)
    rtol_236760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 46), 'rtol', False)
    keyword_236761 = rtol_236760
    kwargs_236762 = {'xtol': keyword_236759, 'rtol': keyword_236761}
    # Getting the type of 'method' (line 100)
    method_236754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'method', False)
    # Calling method(args, kwargs) (line 100)
    method_call_result_236763 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), method_236754, *[f_236755, int_236756, int_236757], **kwargs_236762)
    
    # Assigning a type to the variable 'res' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'res', method_call_result_236763)
    
    # Call to assert_allclose(...): (line 101)
    # Processing the call arguments (line 101)
    float_236765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'float')
    # Getting the type of 'res' (line 101)
    res_236766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'res', False)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'atol' (line 101)
    atol_236767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'atol', False)
    keyword_236768 = atol_236767
    # Getting the type of 'rtol' (line 101)
    rtol_236769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'rtol', False)
    keyword_236770 = rtol_236769
    kwargs_236771 = {'rtol': keyword_236770, 'atol': keyword_236768}
    # Getting the type of 'assert_allclose' (line 101)
    assert_allclose_236764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 101)
    assert_allclose_call_result_236772 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assert_allclose_236764, *[float_236765, res_236766], **kwargs_236771)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_gh_5557(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gh_5557' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_236773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_236773)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gh_5557'
    return stypy_return_type_236773

# Assigning a type to the variable 'test_gh_5557' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'test_gh_5557', test_gh_5557)
# Declaration of the 'TestRootResults' class

class TestRootResults:

    @norecursion
    def test_repr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_repr'
        module_type_store = module_type_store.open_function_context('test_repr', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRootResults.test_repr.__dict__.__setitem__('stypy_localization', localization)
        TestRootResults.test_repr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRootResults.test_repr.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRootResults.test_repr.__dict__.__setitem__('stypy_function_name', 'TestRootResults.test_repr')
        TestRootResults.test_repr.__dict__.__setitem__('stypy_param_names_list', [])
        TestRootResults.test_repr.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRootResults.test_repr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRootResults.test_repr.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRootResults.test_repr.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRootResults.test_repr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRootResults.test_repr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRootResults.test_repr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_repr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_repr(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to RootResults(...): (line 106)
        # Processing the call keyword arguments (line 106)
        float_236776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'float')
        keyword_236777 = float_236776
        int_236778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 41), 'int')
        keyword_236779 = int_236778
        int_236780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 45), 'int')
        keyword_236781 = int_236780
        int_236782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'int')
        keyword_236783 = int_236782
        kwargs_236784 = {'function_calls': keyword_236781, 'flag': keyword_236783, 'root': keyword_236777, 'iterations': keyword_236779}
        # Getting the type of 'zeros' (line 106)
        zeros_236774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'zeros', False)
        # Obtaining the member 'RootResults' of a type (line 106)
        RootResults_236775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), zeros_236774, 'RootResults')
        # Calling RootResults(args, kwargs) (line 106)
        RootResults_call_result_236785 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), RootResults_236775, *[], **kwargs_236784)
        
        # Assigning a type to the variable 'r' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'r', RootResults_call_result_236785)
        
        # Assigning a Str to a Name (line 110):
        
        # Assigning a Str to a Name (line 110):
        str_236786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'str', "      converged: True\n           flag: 'converged'\n function_calls: 46\n     iterations: 44\n           root: 1.0")
        # Assigning a type to the variable 'expected_repr' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'expected_repr', str_236786)
        
        # Call to assert_equal(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to repr(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'r' (line 113)
        r_236789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'r', False)
        # Processing the call keyword arguments (line 113)
        kwargs_236790 = {}
        # Getting the type of 'repr' (line 113)
        repr_236788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'repr', False)
        # Calling repr(args, kwargs) (line 113)
        repr_call_result_236791 = invoke(stypy.reporting.localization.Localization(__file__, 113, 21), repr_236788, *[r_236789], **kwargs_236790)
        
        # Getting the type of 'expected_repr' (line 113)
        expected_repr_236792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'expected_repr', False)
        # Processing the call keyword arguments (line 113)
        kwargs_236793 = {}
        # Getting the type of 'assert_equal' (line 113)
        assert_equal_236787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 113)
        assert_equal_call_result_236794 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_equal_236787, *[repr_call_result_236791, expected_repr_236792], **kwargs_236793)
        
        
        # ################# End of 'test_repr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_repr' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_236795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_repr'
        return stypy_return_type_236795


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 104, 0, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRootResults.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestRootResults' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'TestRootResults', TestRootResults)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
