
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit tests for Krylov space trust-region subproblem solver.
3: 
4: To run it in its simplest form::
5:   nosetests test_optimize.py
6: 
7: '''
8: from __future__ import division, print_function, absolute_import
9: 
10: import numpy as np
11: from scipy.optimize._trlib import (get_trlib_quadratic_subproblem)
12: from numpy.testing import (assert_, assert_array_equal,
13:                            assert_almost_equal,
14:                            assert_equal, assert_array_almost_equal,
15:                            assert_array_less)
16: 
17: KrylovQP = get_trlib_quadratic_subproblem(tol_rel_i=1e-8, tol_rel_b=1e-6)
18: KrylovQP_disp = get_trlib_quadratic_subproblem(tol_rel_i=1e-8, tol_rel_b=1e-6, disp=True)
19: 
20: class TestKrylovQuadraticSubproblem(object):
21: 
22:     def test_for_the_easy_case(self):
23: 
24:         # `H` is chosen such that `g` is not orthogonal to the
25:         # eigenvector associated with the smallest eigenvalue.
26:         H = np.array([[1.0, 0.0, 4.0],
27:                       [0.0, 2.0, 0.0],
28:                       [4.0, 0.0, 3.0]])
29:         g = np.array([5.0, 0.0, 4.0])
30: 
31:         # Trust Radius
32:         trust_radius = 1.0
33: 
34:         # Solve Subproblem
35:         subprob = KrylovQP(x=0,
36:                            fun=lambda x: 0,
37:                            jac=lambda x: g,
38:                            hess=lambda x: None,
39:                            hessp=lambda x, y: H.dot(y))
40:         p, hits_boundary = subprob.solve(trust_radius)
41: 
42:         assert_array_almost_equal(p, np.array([-1.0, 0.0, 0.0]))
43:         assert_equal(hits_boundary, True)
44:         # check kkt satisfaction
45:         assert_almost_equal(
46:                 np.linalg.norm(H.dot(p) + subprob.lam * p + g),
47:                 0.0)
48:         # check trust region constraint
49:         assert_almost_equal(np.linalg.norm(p), trust_radius)
50: 
51:         trust_radius = 0.5
52:         p, hits_boundary = subprob.solve(trust_radius)
53: 
54:         assert_array_almost_equal(p,
55:                 np.array([-0.46125446, 0., -0.19298788]))
56:         assert_equal(hits_boundary, True)
57:         # check kkt satisfaction
58:         assert_almost_equal(
59:                 np.linalg.norm(H.dot(p) + subprob.lam * p + g),
60:                 0.0)
61:         # check trust region constraint
62:         assert_almost_equal(np.linalg.norm(p), trust_radius)
63: 
64:     def test_for_the_hard_case(self):
65: 
66:         # `H` is chosen such that `g` is orthogonal to the
67:         # eigenvector associated with the smallest eigenvalue.
68:         H = np.array([[1.0, 0.0, 4.0],
69:                       [0.0, 2.0, 0.0],
70:                       [4.0, 0.0, 3.0]])
71:         g = np.array([0.0, 2.0, 0.0])
72: 
73:         # Trust Radius
74:         trust_radius = 1.0
75: 
76:         # Solve Subproblem
77:         subprob = KrylovQP(x=0,
78:                            fun=lambda x: 0,
79:                            jac=lambda x: g,
80:                            hess=lambda x: None,
81:                            hessp=lambda x, y: H.dot(y))
82:         p, hits_boundary = subprob.solve(trust_radius)
83: 
84:         assert_array_almost_equal(p, np.array([0.0, -1.0, 0.0]))
85:         # check kkt satisfaction
86:         assert_almost_equal(
87:                 np.linalg.norm(H.dot(p) + subprob.lam * p + g),
88:                 0.0)
89:         # check trust region constraint
90:         assert_almost_equal(np.linalg.norm(p), trust_radius)
91: 
92:         trust_radius = 0.5
93:         p, hits_boundary = subprob.solve(trust_radius)
94: 
95:         assert_array_almost_equal(p, np.array([0.0, -0.5, 0.0]))
96:         # check kkt satisfaction
97:         assert_almost_equal(
98:                 np.linalg.norm(H.dot(p) + subprob.lam * p + g),
99:                 0.0)
100:         # check trust region constraint
101:         assert_almost_equal(np.linalg.norm(p), trust_radius)
102: 
103:     def test_for_interior_convergence(self):
104: 
105:         H = np.array([[1.812159, 0.82687265, 0.21838879, -0.52487006, 0.25436988],
106:                       [0.82687265, 2.66380283, 0.31508988, -0.40144163, 0.08811588],
107:                       [0.21838879, 0.31508988, 2.38020726, -0.3166346, 0.27363867],
108:                       [-0.52487006, -0.40144163, -0.3166346, 1.61927182, -0.42140166],
109:                       [0.25436988, 0.08811588, 0.27363867, -0.42140166, 1.33243101]])
110:         g = np.array([0.75798952, 0.01421945, 0.33847612, 0.83725004, -0.47909534])
111:         trust_radius = 1.1
112: 
113:         # Solve Subproblem
114:         subprob = KrylovQP(x=0,
115:                            fun=lambda x: 0,
116:                            jac=lambda x: g,
117:                            hess=lambda x: None,
118:                            hessp=lambda x, y: H.dot(y))
119:         p, hits_boundary = subprob.solve(trust_radius)
120: 
121:         # check kkt satisfaction
122:         assert_almost_equal(
123:                 np.linalg.norm(H.dot(p) + subprob.lam * p + g),
124:                 0.0)
125: 
126:         assert_array_almost_equal(p, [-0.68585435, 0.1222621, -0.22090999,
127:                                       -0.67005053, 0.31586769])
128:         assert_array_almost_equal(hits_boundary, False)
129: 
130:     def test_for_very_close_to_zero(self):
131: 
132:         H = np.array([[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809],
133:                       [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396],
134:                       [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957],
135:                       [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298],
136:                       [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]])
137:         g = np.array([0, 0, 0, 0, 1e-6])
138:         trust_radius = 1.1
139: 
140:         # Solve Subproblem
141:         subprob = KrylovQP(x=0,
142:                            fun=lambda x: 0,
143:                            jac=lambda x: g,
144:                            hess=lambda x: None,
145:                            hessp=lambda x, y: H.dot(y))
146:         p, hits_boundary = subprob.solve(trust_radius)
147: 
148:         # check kkt satisfaction
149:         assert_almost_equal(
150:                 np.linalg.norm(H.dot(p) + subprob.lam * p + g),
151:                 0.0)
152:         # check trust region constraint
153:         assert_almost_equal(np.linalg.norm(p), trust_radius)
154: 
155:         assert_array_almost_equal(p, [0.06910534, -0.01432721,
156:                                       -0.65311947, -0.23815972,
157:                                       -0.84954934])
158:         assert_array_almost_equal(hits_boundary, True)
159: 
160:     def test_disp(self, capsys):
161:         H = -np.eye(5)
162:         g = np.array([0, 0, 0, 0, 1e-6])
163:         trust_radius = 1.1
164: 
165:         subprob = KrylovQP_disp(x=0,
166:                                 fun=lambda x: 0,
167:                                 jac=lambda x: g,
168:                                 hess=lambda x: None,
169:                                 hessp=lambda x, y: H.dot(y))
170:         p, hits_boundary = subprob.solve(trust_radius)
171:         out, err = capsys.readouterr()
172:         assert_(out.startswith(' TR Solving trust region problem'), repr(out))
173: 
174: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_235674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nUnit tests for Krylov space trust-region subproblem solver.\n\nTo run it in its simplest form::\n  nosetests test_optimize.py\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_235675 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_235675) is not StypyTypeError):

    if (import_235675 != 'pyd_module'):
        __import__(import_235675)
        sys_modules_235676 = sys.modules[import_235675]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_235676.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_235675)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize._trlib import get_trlib_quadratic_subproblem' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_235677 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._trlib')

if (type(import_235677) is not StypyTypeError):

    if (import_235677 != 'pyd_module'):
        __import__(import_235677)
        sys_modules_235678 = sys.modules[import_235677]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._trlib', sys_modules_235678.module_type_store, module_type_store, ['get_trlib_quadratic_subproblem'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_235678, sys_modules_235678.module_type_store, module_type_store)
    else:
        from scipy.optimize._trlib import get_trlib_quadratic_subproblem

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._trlib', None, module_type_store, ['get_trlib_quadratic_subproblem'], [get_trlib_quadratic_subproblem])

else:
    # Assigning a type to the variable 'scipy.optimize._trlib' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._trlib', import_235677)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.testing import assert_, assert_array_equal, assert_almost_equal, assert_equal, assert_array_almost_equal, assert_array_less' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_235679 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing')

if (type(import_235679) is not StypyTypeError):

    if (import_235679 != 'pyd_module'):
        __import__(import_235679)
        sys_modules_235680 = sys.modules[import_235679]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', sys_modules_235680.module_type_store, module_type_store, ['assert_', 'assert_array_equal', 'assert_almost_equal', 'assert_equal', 'assert_array_almost_equal', 'assert_array_less'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_235680, sys_modules_235680.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_array_equal, assert_almost_equal, assert_equal, assert_array_almost_equal, assert_array_less

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_array_equal', 'assert_almost_equal', 'assert_equal', 'assert_array_almost_equal', 'assert_array_less'], [assert_, assert_array_equal, assert_almost_equal, assert_equal, assert_array_almost_equal, assert_array_less])

else:
    # Assigning a type to the variable 'numpy.testing' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', import_235679)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


# Assigning a Call to a Name (line 17):

# Assigning a Call to a Name (line 17):

# Call to get_trlib_quadratic_subproblem(...): (line 17)
# Processing the call keyword arguments (line 17)
float_235682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 52), 'float')
keyword_235683 = float_235682
float_235684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 68), 'float')
keyword_235685 = float_235684
kwargs_235686 = {'tol_rel_i': keyword_235683, 'tol_rel_b': keyword_235685}
# Getting the type of 'get_trlib_quadratic_subproblem' (line 17)
get_trlib_quadratic_subproblem_235681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'get_trlib_quadratic_subproblem', False)
# Calling get_trlib_quadratic_subproblem(args, kwargs) (line 17)
get_trlib_quadratic_subproblem_call_result_235687 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), get_trlib_quadratic_subproblem_235681, *[], **kwargs_235686)

# Assigning a type to the variable 'KrylovQP' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'KrylovQP', get_trlib_quadratic_subproblem_call_result_235687)

# Assigning a Call to a Name (line 18):

# Assigning a Call to a Name (line 18):

# Call to get_trlib_quadratic_subproblem(...): (line 18)
# Processing the call keyword arguments (line 18)
float_235689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 57), 'float')
keyword_235690 = float_235689
float_235691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 73), 'float')
keyword_235692 = float_235691
# Getting the type of 'True' (line 18)
True_235693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 84), 'True', False)
keyword_235694 = True_235693
kwargs_235695 = {'tol_rel_i': keyword_235690, 'disp': keyword_235694, 'tol_rel_b': keyword_235692}
# Getting the type of 'get_trlib_quadratic_subproblem' (line 18)
get_trlib_quadratic_subproblem_235688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'get_trlib_quadratic_subproblem', False)
# Calling get_trlib_quadratic_subproblem(args, kwargs) (line 18)
get_trlib_quadratic_subproblem_call_result_235696 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), get_trlib_quadratic_subproblem_235688, *[], **kwargs_235695)

# Assigning a type to the variable 'KrylovQP_disp' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'KrylovQP_disp', get_trlib_quadratic_subproblem_call_result_235696)
# Declaration of the 'TestKrylovQuadraticSubproblem' class

class TestKrylovQuadraticSubproblem(object, ):

    @norecursion
    def test_for_the_easy_case(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_the_easy_case'
        module_type_store = module_type_store.open_function_context('test_for_the_easy_case', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_localization', localization)
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_function_name', 'TestKrylovQuadraticSubproblem.test_for_the_easy_case')
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_param_names_list', [])
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKrylovQuadraticSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKrylovQuadraticSubproblem.test_for_the_easy_case', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_the_easy_case', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_the_easy_case(...)' code ##################

        
        # Assigning a Call to a Name (line 26):
        
        # Assigning a Call to a Name (line 26):
        
        # Call to array(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_235699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_235700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        float_235701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 22), list_235700, float_235701)
        # Adding element type (line 26)
        float_235702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 22), list_235700, float_235702)
        # Adding element type (line 26)
        float_235703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 22), list_235700, float_235703)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_235699, list_235700)
        # Adding element type (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_235704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        float_235705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), list_235704, float_235705)
        # Adding element type (line 27)
        float_235706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), list_235704, float_235706)
        # Adding element type (line 27)
        float_235707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), list_235704, float_235707)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_235699, list_235704)
        # Adding element type (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_235708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        float_235709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 22), list_235708, float_235709)
        # Adding element type (line 28)
        float_235710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 22), list_235708, float_235710)
        # Adding element type (line 28)
        float_235711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 22), list_235708, float_235711)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_235699, list_235708)
        
        # Processing the call keyword arguments (line 26)
        kwargs_235712 = {}
        # Getting the type of 'np' (line 26)
        np_235697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 26)
        array_235698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), np_235697, 'array')
        # Calling array(args, kwargs) (line 26)
        array_call_result_235713 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), array_235698, *[list_235699], **kwargs_235712)
        
        # Assigning a type to the variable 'H' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'H', array_call_result_235713)
        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Call to array(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_235716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        float_235717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_235716, float_235717)
        # Adding element type (line 29)
        float_235718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_235716, float_235718)
        # Adding element type (line 29)
        float_235719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_235716, float_235719)
        
        # Processing the call keyword arguments (line 29)
        kwargs_235720 = {}
        # Getting the type of 'np' (line 29)
        np_235714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 29)
        array_235715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), np_235714, 'array')
        # Calling array(args, kwargs) (line 29)
        array_call_result_235721 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), array_235715, *[list_235716], **kwargs_235720)
        
        # Assigning a type to the variable 'g' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'g', array_call_result_235721)
        
        # Assigning a Num to a Name (line 32):
        
        # Assigning a Num to a Name (line 32):
        float_235722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'float')
        # Assigning a type to the variable 'trust_radius' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'trust_radius', float_235722)
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to KrylovQP(...): (line 35)
        # Processing the call keyword arguments (line 35)
        int_235724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 29), 'int')
        keyword_235725 = int_235724

        @norecursion
        def _stypy_temp_lambda_137(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_137'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_137', 36, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_137.stypy_localization = localization
            _stypy_temp_lambda_137.stypy_type_of_self = None
            _stypy_temp_lambda_137.stypy_type_store = module_type_store
            _stypy_temp_lambda_137.stypy_function_name = '_stypy_temp_lambda_137'
            _stypy_temp_lambda_137.stypy_param_names_list = ['x']
            _stypy_temp_lambda_137.stypy_varargs_param_name = None
            _stypy_temp_lambda_137.stypy_kwargs_param_name = None
            _stypy_temp_lambda_137.stypy_call_defaults = defaults
            _stypy_temp_lambda_137.stypy_call_varargs = varargs
            _stypy_temp_lambda_137.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_137', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_137', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_235726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 41), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'stypy_return_type', int_235726)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_137' in the type store
            # Getting the type of 'stypy_return_type' (line 36)
            stypy_return_type_235727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235727)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_137'
            return stypy_return_type_235727

        # Assigning a type to the variable '_stypy_temp_lambda_137' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), '_stypy_temp_lambda_137', _stypy_temp_lambda_137)
        # Getting the type of '_stypy_temp_lambda_137' (line 36)
        _stypy_temp_lambda_137_235728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), '_stypy_temp_lambda_137')
        keyword_235729 = _stypy_temp_lambda_137_235728

        @norecursion
        def _stypy_temp_lambda_138(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_138'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_138', 37, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_138.stypy_localization = localization
            _stypy_temp_lambda_138.stypy_type_of_self = None
            _stypy_temp_lambda_138.stypy_type_store = module_type_store
            _stypy_temp_lambda_138.stypy_function_name = '_stypy_temp_lambda_138'
            _stypy_temp_lambda_138.stypy_param_names_list = ['x']
            _stypy_temp_lambda_138.stypy_varargs_param_name = None
            _stypy_temp_lambda_138.stypy_kwargs_param_name = None
            _stypy_temp_lambda_138.stypy_call_defaults = defaults
            _stypy_temp_lambda_138.stypy_call_varargs = varargs
            _stypy_temp_lambda_138.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_138', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_138', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'g' (line 37)
            g_235730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 41), 'g', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'stypy_return_type', g_235730)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_138' in the type store
            # Getting the type of 'stypy_return_type' (line 37)
            stypy_return_type_235731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235731)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_138'
            return stypy_return_type_235731

        # Assigning a type to the variable '_stypy_temp_lambda_138' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), '_stypy_temp_lambda_138', _stypy_temp_lambda_138)
        # Getting the type of '_stypy_temp_lambda_138' (line 37)
        _stypy_temp_lambda_138_235732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), '_stypy_temp_lambda_138')
        keyword_235733 = _stypy_temp_lambda_138_235732

        @norecursion
        def _stypy_temp_lambda_139(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_139'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_139', 38, 32, True)
            # Passed parameters checking function
            _stypy_temp_lambda_139.stypy_localization = localization
            _stypy_temp_lambda_139.stypy_type_of_self = None
            _stypy_temp_lambda_139.stypy_type_store = module_type_store
            _stypy_temp_lambda_139.stypy_function_name = '_stypy_temp_lambda_139'
            _stypy_temp_lambda_139.stypy_param_names_list = ['x']
            _stypy_temp_lambda_139.stypy_varargs_param_name = None
            _stypy_temp_lambda_139.stypy_kwargs_param_name = None
            _stypy_temp_lambda_139.stypy_call_defaults = defaults
            _stypy_temp_lambda_139.stypy_call_varargs = varargs
            _stypy_temp_lambda_139.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_139', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_139', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 38)
            None_235734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'stypy_return_type', None_235734)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_139' in the type store
            # Getting the type of 'stypy_return_type' (line 38)
            stypy_return_type_235735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235735)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_139'
            return stypy_return_type_235735

        # Assigning a type to the variable '_stypy_temp_lambda_139' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), '_stypy_temp_lambda_139', _stypy_temp_lambda_139)
        # Getting the type of '_stypy_temp_lambda_139' (line 38)
        _stypy_temp_lambda_139_235736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), '_stypy_temp_lambda_139')
        keyword_235737 = _stypy_temp_lambda_139_235736

        @norecursion
        def _stypy_temp_lambda_140(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_140'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_140', 39, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_140.stypy_localization = localization
            _stypy_temp_lambda_140.stypy_type_of_self = None
            _stypy_temp_lambda_140.stypy_type_store = module_type_store
            _stypy_temp_lambda_140.stypy_function_name = '_stypy_temp_lambda_140'
            _stypy_temp_lambda_140.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_140.stypy_varargs_param_name = None
            _stypy_temp_lambda_140.stypy_kwargs_param_name = None
            _stypy_temp_lambda_140.stypy_call_defaults = defaults
            _stypy_temp_lambda_140.stypy_call_varargs = varargs
            _stypy_temp_lambda_140.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_140', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_140', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to dot(...): (line 39)
            # Processing the call arguments (line 39)
            # Getting the type of 'y' (line 39)
            y_235740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 52), 'y', False)
            # Processing the call keyword arguments (line 39)
            kwargs_235741 = {}
            # Getting the type of 'H' (line 39)
            H_235738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 46), 'H', False)
            # Obtaining the member 'dot' of a type (line 39)
            dot_235739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 46), H_235738, 'dot')
            # Calling dot(args, kwargs) (line 39)
            dot_call_result_235742 = invoke(stypy.reporting.localization.Localization(__file__, 39, 46), dot_235739, *[y_235740], **kwargs_235741)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'stypy_return_type', dot_call_result_235742)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_140' in the type store
            # Getting the type of 'stypy_return_type' (line 39)
            stypy_return_type_235743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235743)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_140'
            return stypy_return_type_235743

        # Assigning a type to the variable '_stypy_temp_lambda_140' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), '_stypy_temp_lambda_140', _stypy_temp_lambda_140)
        # Getting the type of '_stypy_temp_lambda_140' (line 39)
        _stypy_temp_lambda_140_235744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), '_stypy_temp_lambda_140')
        keyword_235745 = _stypy_temp_lambda_140_235744
        kwargs_235746 = {'fun': keyword_235729, 'x': keyword_235725, 'hess': keyword_235737, 'jac': keyword_235733, 'hessp': keyword_235745}
        # Getting the type of 'KrylovQP' (line 35)
        KrylovQP_235723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'KrylovQP', False)
        # Calling KrylovQP(args, kwargs) (line 35)
        KrylovQP_call_result_235747 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), KrylovQP_235723, *[], **kwargs_235746)
        
        # Assigning a type to the variable 'subprob' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'subprob', KrylovQP_call_result_235747)
        
        # Assigning a Call to a Tuple (line 40):
        
        # Assigning a Subscript to a Name (line 40):
        
        # Obtaining the type of the subscript
        int_235748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'int')
        
        # Call to solve(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'trust_radius' (line 40)
        trust_radius_235751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 40)
        kwargs_235752 = {}
        # Getting the type of 'subprob' (line 40)
        subprob_235749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 40)
        solve_235750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), subprob_235749, 'solve')
        # Calling solve(args, kwargs) (line 40)
        solve_call_result_235753 = invoke(stypy.reporting.localization.Localization(__file__, 40, 27), solve_235750, *[trust_radius_235751], **kwargs_235752)
        
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___235754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), solve_call_result_235753, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_235755 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), getitem___235754, int_235748)
        
        # Assigning a type to the variable 'tuple_var_assignment_235658' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_235658', subscript_call_result_235755)
        
        # Assigning a Subscript to a Name (line 40):
        
        # Obtaining the type of the subscript
        int_235756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'int')
        
        # Call to solve(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'trust_radius' (line 40)
        trust_radius_235759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 40)
        kwargs_235760 = {}
        # Getting the type of 'subprob' (line 40)
        subprob_235757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 40)
        solve_235758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), subprob_235757, 'solve')
        # Calling solve(args, kwargs) (line 40)
        solve_call_result_235761 = invoke(stypy.reporting.localization.Localization(__file__, 40, 27), solve_235758, *[trust_radius_235759], **kwargs_235760)
        
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___235762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), solve_call_result_235761, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_235763 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), getitem___235762, int_235756)
        
        # Assigning a type to the variable 'tuple_var_assignment_235659' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_235659', subscript_call_result_235763)
        
        # Assigning a Name to a Name (line 40):
        # Getting the type of 'tuple_var_assignment_235658' (line 40)
        tuple_var_assignment_235658_235764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_235658')
        # Assigning a type to the variable 'p' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'p', tuple_var_assignment_235658_235764)
        
        # Assigning a Name to a Name (line 40):
        # Getting the type of 'tuple_var_assignment_235659' (line 40)
        tuple_var_assignment_235659_235765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_235659')
        # Assigning a type to the variable 'hits_boundary' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'hits_boundary', tuple_var_assignment_235659_235765)
        
        # Call to assert_array_almost_equal(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'p' (line 42)
        p_235767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 34), 'p', False)
        
        # Call to array(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_235770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        float_235771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 46), list_235770, float_235771)
        # Adding element type (line 42)
        float_235772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 46), list_235770, float_235772)
        # Adding element type (line 42)
        float_235773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 46), list_235770, float_235773)
        
        # Processing the call keyword arguments (line 42)
        kwargs_235774 = {}
        # Getting the type of 'np' (line 42)
        np_235768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 37), 'np', False)
        # Obtaining the member 'array' of a type (line 42)
        array_235769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 37), np_235768, 'array')
        # Calling array(args, kwargs) (line 42)
        array_call_result_235775 = invoke(stypy.reporting.localization.Localization(__file__, 42, 37), array_235769, *[list_235770], **kwargs_235774)
        
        # Processing the call keyword arguments (line 42)
        kwargs_235776 = {}
        # Getting the type of 'assert_array_almost_equal' (line 42)
        assert_array_almost_equal_235766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 42)
        assert_array_almost_equal_call_result_235777 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_array_almost_equal_235766, *[p_235767, array_call_result_235775], **kwargs_235776)
        
        
        # Call to assert_equal(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'hits_boundary' (line 43)
        hits_boundary_235779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'hits_boundary', False)
        # Getting the type of 'True' (line 43)
        True_235780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'True', False)
        # Processing the call keyword arguments (line 43)
        kwargs_235781 = {}
        # Getting the type of 'assert_equal' (line 43)
        assert_equal_235778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 43)
        assert_equal_call_result_235782 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), assert_equal_235778, *[hits_boundary_235779, True_235780], **kwargs_235781)
        
        
        # Call to assert_almost_equal(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to norm(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to dot(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'p' (line 46)
        p_235789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'p', False)
        # Processing the call keyword arguments (line 46)
        kwargs_235790 = {}
        # Getting the type of 'H' (line 46)
        H_235787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'H', False)
        # Obtaining the member 'dot' of a type (line 46)
        dot_235788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 31), H_235787, 'dot')
        # Calling dot(args, kwargs) (line 46)
        dot_call_result_235791 = invoke(stypy.reporting.localization.Localization(__file__, 46, 31), dot_235788, *[p_235789], **kwargs_235790)
        
        # Getting the type of 'subprob' (line 46)
        subprob_235792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'subprob', False)
        # Obtaining the member 'lam' of a type (line 46)
        lam_235793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 42), subprob_235792, 'lam')
        # Getting the type of 'p' (line 46)
        p_235794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 56), 'p', False)
        # Applying the binary operator '*' (line 46)
        result_mul_235795 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 42), '*', lam_235793, p_235794)
        
        # Applying the binary operator '+' (line 46)
        result_add_235796 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 31), '+', dot_call_result_235791, result_mul_235795)
        
        # Getting the type of 'g' (line 46)
        g_235797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 60), 'g', False)
        # Applying the binary operator '+' (line 46)
        result_add_235798 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 58), '+', result_add_235796, g_235797)
        
        # Processing the call keyword arguments (line 46)
        kwargs_235799 = {}
        # Getting the type of 'np' (line 46)
        np_235784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 46)
        linalg_235785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), np_235784, 'linalg')
        # Obtaining the member 'norm' of a type (line 46)
        norm_235786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), linalg_235785, 'norm')
        # Calling norm(args, kwargs) (line 46)
        norm_call_result_235800 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), norm_235786, *[result_add_235798], **kwargs_235799)
        
        float_235801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 16), 'float')
        # Processing the call keyword arguments (line 45)
        kwargs_235802 = {}
        # Getting the type of 'assert_almost_equal' (line 45)
        assert_almost_equal_235783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 45)
        assert_almost_equal_call_result_235803 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assert_almost_equal_235783, *[norm_call_result_235800, float_235801], **kwargs_235802)
        
        
        # Call to assert_almost_equal(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to norm(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'p' (line 49)
        p_235808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'p', False)
        # Processing the call keyword arguments (line 49)
        kwargs_235809 = {}
        # Getting the type of 'np' (line 49)
        np_235805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'np', False)
        # Obtaining the member 'linalg' of a type (line 49)
        linalg_235806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 28), np_235805, 'linalg')
        # Obtaining the member 'norm' of a type (line 49)
        norm_235807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 28), linalg_235806, 'norm')
        # Calling norm(args, kwargs) (line 49)
        norm_call_result_235810 = invoke(stypy.reporting.localization.Localization(__file__, 49, 28), norm_235807, *[p_235808], **kwargs_235809)
        
        # Getting the type of 'trust_radius' (line 49)
        trust_radius_235811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 47), 'trust_radius', False)
        # Processing the call keyword arguments (line 49)
        kwargs_235812 = {}
        # Getting the type of 'assert_almost_equal' (line 49)
        assert_almost_equal_235804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 49)
        assert_almost_equal_call_result_235813 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assert_almost_equal_235804, *[norm_call_result_235810, trust_radius_235811], **kwargs_235812)
        
        
        # Assigning a Num to a Name (line 51):
        
        # Assigning a Num to a Name (line 51):
        float_235814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'float')
        # Assigning a type to the variable 'trust_radius' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'trust_radius', float_235814)
        
        # Assigning a Call to a Tuple (line 52):
        
        # Assigning a Subscript to a Name (line 52):
        
        # Obtaining the type of the subscript
        int_235815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'int')
        
        # Call to solve(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'trust_radius' (line 52)
        trust_radius_235818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 52)
        kwargs_235819 = {}
        # Getting the type of 'subprob' (line 52)
        subprob_235816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 52)
        solve_235817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 27), subprob_235816, 'solve')
        # Calling solve(args, kwargs) (line 52)
        solve_call_result_235820 = invoke(stypy.reporting.localization.Localization(__file__, 52, 27), solve_235817, *[trust_radius_235818], **kwargs_235819)
        
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___235821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), solve_call_result_235820, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_235822 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), getitem___235821, int_235815)
        
        # Assigning a type to the variable 'tuple_var_assignment_235660' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_235660', subscript_call_result_235822)
        
        # Assigning a Subscript to a Name (line 52):
        
        # Obtaining the type of the subscript
        int_235823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'int')
        
        # Call to solve(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'trust_radius' (line 52)
        trust_radius_235826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 52)
        kwargs_235827 = {}
        # Getting the type of 'subprob' (line 52)
        subprob_235824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 52)
        solve_235825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 27), subprob_235824, 'solve')
        # Calling solve(args, kwargs) (line 52)
        solve_call_result_235828 = invoke(stypy.reporting.localization.Localization(__file__, 52, 27), solve_235825, *[trust_radius_235826], **kwargs_235827)
        
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___235829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), solve_call_result_235828, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_235830 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), getitem___235829, int_235823)
        
        # Assigning a type to the variable 'tuple_var_assignment_235661' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_235661', subscript_call_result_235830)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'tuple_var_assignment_235660' (line 52)
        tuple_var_assignment_235660_235831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_235660')
        # Assigning a type to the variable 'p' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'p', tuple_var_assignment_235660_235831)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'tuple_var_assignment_235661' (line 52)
        tuple_var_assignment_235661_235832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_235661')
        # Assigning a type to the variable 'hits_boundary' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'hits_boundary', tuple_var_assignment_235661_235832)
        
        # Call to assert_array_almost_equal(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'p' (line 54)
        p_235834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'p', False)
        
        # Call to array(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_235837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        float_235838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 25), list_235837, float_235838)
        # Adding element type (line 55)
        float_235839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 25), list_235837, float_235839)
        # Adding element type (line 55)
        float_235840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 25), list_235837, float_235840)
        
        # Processing the call keyword arguments (line 55)
        kwargs_235841 = {}
        # Getting the type of 'np' (line 55)
        np_235835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 55)
        array_235836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), np_235835, 'array')
        # Calling array(args, kwargs) (line 55)
        array_call_result_235842 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), array_235836, *[list_235837], **kwargs_235841)
        
        # Processing the call keyword arguments (line 54)
        kwargs_235843 = {}
        # Getting the type of 'assert_array_almost_equal' (line 54)
        assert_array_almost_equal_235833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 54)
        assert_array_almost_equal_call_result_235844 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert_array_almost_equal_235833, *[p_235834, array_call_result_235842], **kwargs_235843)
        
        
        # Call to assert_equal(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'hits_boundary' (line 56)
        hits_boundary_235846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'hits_boundary', False)
        # Getting the type of 'True' (line 56)
        True_235847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'True', False)
        # Processing the call keyword arguments (line 56)
        kwargs_235848 = {}
        # Getting the type of 'assert_equal' (line 56)
        assert_equal_235845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 56)
        assert_equal_call_result_235849 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assert_equal_235845, *[hits_boundary_235846, True_235847], **kwargs_235848)
        
        
        # Call to assert_almost_equal(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to norm(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Call to dot(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'p' (line 59)
        p_235856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'p', False)
        # Processing the call keyword arguments (line 59)
        kwargs_235857 = {}
        # Getting the type of 'H' (line 59)
        H_235854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'H', False)
        # Obtaining the member 'dot' of a type (line 59)
        dot_235855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 31), H_235854, 'dot')
        # Calling dot(args, kwargs) (line 59)
        dot_call_result_235858 = invoke(stypy.reporting.localization.Localization(__file__, 59, 31), dot_235855, *[p_235856], **kwargs_235857)
        
        # Getting the type of 'subprob' (line 59)
        subprob_235859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 42), 'subprob', False)
        # Obtaining the member 'lam' of a type (line 59)
        lam_235860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 42), subprob_235859, 'lam')
        # Getting the type of 'p' (line 59)
        p_235861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 56), 'p', False)
        # Applying the binary operator '*' (line 59)
        result_mul_235862 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 42), '*', lam_235860, p_235861)
        
        # Applying the binary operator '+' (line 59)
        result_add_235863 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 31), '+', dot_call_result_235858, result_mul_235862)
        
        # Getting the type of 'g' (line 59)
        g_235864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 60), 'g', False)
        # Applying the binary operator '+' (line 59)
        result_add_235865 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 58), '+', result_add_235863, g_235864)
        
        # Processing the call keyword arguments (line 59)
        kwargs_235866 = {}
        # Getting the type of 'np' (line 59)
        np_235851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 59)
        linalg_235852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), np_235851, 'linalg')
        # Obtaining the member 'norm' of a type (line 59)
        norm_235853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), linalg_235852, 'norm')
        # Calling norm(args, kwargs) (line 59)
        norm_call_result_235867 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), norm_235853, *[result_add_235865], **kwargs_235866)
        
        float_235868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'float')
        # Processing the call keyword arguments (line 58)
        kwargs_235869 = {}
        # Getting the type of 'assert_almost_equal' (line 58)
        assert_almost_equal_235850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 58)
        assert_almost_equal_call_result_235870 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_almost_equal_235850, *[norm_call_result_235867, float_235868], **kwargs_235869)
        
        
        # Call to assert_almost_equal(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to norm(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'p' (line 62)
        p_235875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 43), 'p', False)
        # Processing the call keyword arguments (line 62)
        kwargs_235876 = {}
        # Getting the type of 'np' (line 62)
        np_235872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'np', False)
        # Obtaining the member 'linalg' of a type (line 62)
        linalg_235873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 28), np_235872, 'linalg')
        # Obtaining the member 'norm' of a type (line 62)
        norm_235874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 28), linalg_235873, 'norm')
        # Calling norm(args, kwargs) (line 62)
        norm_call_result_235877 = invoke(stypy.reporting.localization.Localization(__file__, 62, 28), norm_235874, *[p_235875], **kwargs_235876)
        
        # Getting the type of 'trust_radius' (line 62)
        trust_radius_235878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'trust_radius', False)
        # Processing the call keyword arguments (line 62)
        kwargs_235879 = {}
        # Getting the type of 'assert_almost_equal' (line 62)
        assert_almost_equal_235871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 62)
        assert_almost_equal_call_result_235880 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_almost_equal_235871, *[norm_call_result_235877, trust_radius_235878], **kwargs_235879)
        
        
        # ################# End of 'test_for_the_easy_case(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_the_easy_case' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_235881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_235881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_the_easy_case'
        return stypy_return_type_235881


    @norecursion
    def test_for_the_hard_case(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_the_hard_case'
        module_type_store = module_type_store.open_function_context('test_for_the_hard_case', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_localization', localization)
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_function_name', 'TestKrylovQuadraticSubproblem.test_for_the_hard_case')
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_param_names_list', [])
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKrylovQuadraticSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKrylovQuadraticSubproblem.test_for_the_hard_case', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_the_hard_case', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_the_hard_case(...)' code ##################

        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to array(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_235884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_235885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        float_235886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_235885, float_235886)
        # Adding element type (line 68)
        float_235887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_235885, float_235887)
        # Adding element type (line 68)
        float_235888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_235885, float_235888)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), list_235884, list_235885)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_235889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        float_235890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 22), list_235889, float_235890)
        # Adding element type (line 69)
        float_235891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 22), list_235889, float_235891)
        # Adding element type (line 69)
        float_235892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 22), list_235889, float_235892)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), list_235884, list_235889)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_235893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        float_235894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), list_235893, float_235894)
        # Adding element type (line 70)
        float_235895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), list_235893, float_235895)
        # Adding element type (line 70)
        float_235896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), list_235893, float_235896)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), list_235884, list_235893)
        
        # Processing the call keyword arguments (line 68)
        kwargs_235897 = {}
        # Getting the type of 'np' (line 68)
        np_235882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 68)
        array_235883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), np_235882, 'array')
        # Calling array(args, kwargs) (line 68)
        array_call_result_235898 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), array_235883, *[list_235884], **kwargs_235897)
        
        # Assigning a type to the variable 'H' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'H', array_call_result_235898)
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to array(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_235901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        float_235902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 21), list_235901, float_235902)
        # Adding element type (line 71)
        float_235903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 21), list_235901, float_235903)
        # Adding element type (line 71)
        float_235904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 21), list_235901, float_235904)
        
        # Processing the call keyword arguments (line 71)
        kwargs_235905 = {}
        # Getting the type of 'np' (line 71)
        np_235899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 71)
        array_235900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), np_235899, 'array')
        # Calling array(args, kwargs) (line 71)
        array_call_result_235906 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), array_235900, *[list_235901], **kwargs_235905)
        
        # Assigning a type to the variable 'g' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'g', array_call_result_235906)
        
        # Assigning a Num to a Name (line 74):
        
        # Assigning a Num to a Name (line 74):
        float_235907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'float')
        # Assigning a type to the variable 'trust_radius' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'trust_radius', float_235907)
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to KrylovQP(...): (line 77)
        # Processing the call keyword arguments (line 77)
        int_235909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 29), 'int')
        keyword_235910 = int_235909

        @norecursion
        def _stypy_temp_lambda_141(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_141'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_141', 78, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_141.stypy_localization = localization
            _stypy_temp_lambda_141.stypy_type_of_self = None
            _stypy_temp_lambda_141.stypy_type_store = module_type_store
            _stypy_temp_lambda_141.stypy_function_name = '_stypy_temp_lambda_141'
            _stypy_temp_lambda_141.stypy_param_names_list = ['x']
            _stypy_temp_lambda_141.stypy_varargs_param_name = None
            _stypy_temp_lambda_141.stypy_kwargs_param_name = None
            _stypy_temp_lambda_141.stypy_call_defaults = defaults
            _stypy_temp_lambda_141.stypy_call_varargs = varargs
            _stypy_temp_lambda_141.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_141', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_141', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_235911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 41), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), 'stypy_return_type', int_235911)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_141' in the type store
            # Getting the type of 'stypy_return_type' (line 78)
            stypy_return_type_235912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235912)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_141'
            return stypy_return_type_235912

        # Assigning a type to the variable '_stypy_temp_lambda_141' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), '_stypy_temp_lambda_141', _stypy_temp_lambda_141)
        # Getting the type of '_stypy_temp_lambda_141' (line 78)
        _stypy_temp_lambda_141_235913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), '_stypy_temp_lambda_141')
        keyword_235914 = _stypy_temp_lambda_141_235913

        @norecursion
        def _stypy_temp_lambda_142(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_142'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_142', 79, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_142.stypy_localization = localization
            _stypy_temp_lambda_142.stypy_type_of_self = None
            _stypy_temp_lambda_142.stypy_type_store = module_type_store
            _stypy_temp_lambda_142.stypy_function_name = '_stypy_temp_lambda_142'
            _stypy_temp_lambda_142.stypy_param_names_list = ['x']
            _stypy_temp_lambda_142.stypy_varargs_param_name = None
            _stypy_temp_lambda_142.stypy_kwargs_param_name = None
            _stypy_temp_lambda_142.stypy_call_defaults = defaults
            _stypy_temp_lambda_142.stypy_call_varargs = varargs
            _stypy_temp_lambda_142.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_142', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_142', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'g' (line 79)
            g_235915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 41), 'g', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'stypy_return_type', g_235915)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_142' in the type store
            # Getting the type of 'stypy_return_type' (line 79)
            stypy_return_type_235916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235916)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_142'
            return stypy_return_type_235916

        # Assigning a type to the variable '_stypy_temp_lambda_142' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), '_stypy_temp_lambda_142', _stypy_temp_lambda_142)
        # Getting the type of '_stypy_temp_lambda_142' (line 79)
        _stypy_temp_lambda_142_235917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), '_stypy_temp_lambda_142')
        keyword_235918 = _stypy_temp_lambda_142_235917

        @norecursion
        def _stypy_temp_lambda_143(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_143'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_143', 80, 32, True)
            # Passed parameters checking function
            _stypy_temp_lambda_143.stypy_localization = localization
            _stypy_temp_lambda_143.stypy_type_of_self = None
            _stypy_temp_lambda_143.stypy_type_store = module_type_store
            _stypy_temp_lambda_143.stypy_function_name = '_stypy_temp_lambda_143'
            _stypy_temp_lambda_143.stypy_param_names_list = ['x']
            _stypy_temp_lambda_143.stypy_varargs_param_name = None
            _stypy_temp_lambda_143.stypy_kwargs_param_name = None
            _stypy_temp_lambda_143.stypy_call_defaults = defaults
            _stypy_temp_lambda_143.stypy_call_varargs = varargs
            _stypy_temp_lambda_143.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_143', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_143', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 80)
            None_235919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 42), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), 'stypy_return_type', None_235919)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_143' in the type store
            # Getting the type of 'stypy_return_type' (line 80)
            stypy_return_type_235920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235920)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_143'
            return stypy_return_type_235920

        # Assigning a type to the variable '_stypy_temp_lambda_143' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), '_stypy_temp_lambda_143', _stypy_temp_lambda_143)
        # Getting the type of '_stypy_temp_lambda_143' (line 80)
        _stypy_temp_lambda_143_235921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), '_stypy_temp_lambda_143')
        keyword_235922 = _stypy_temp_lambda_143_235921

        @norecursion
        def _stypy_temp_lambda_144(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_144'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_144', 81, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_144.stypy_localization = localization
            _stypy_temp_lambda_144.stypy_type_of_self = None
            _stypy_temp_lambda_144.stypy_type_store = module_type_store
            _stypy_temp_lambda_144.stypy_function_name = '_stypy_temp_lambda_144'
            _stypy_temp_lambda_144.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_144.stypy_varargs_param_name = None
            _stypy_temp_lambda_144.stypy_kwargs_param_name = None
            _stypy_temp_lambda_144.stypy_call_defaults = defaults
            _stypy_temp_lambda_144.stypy_call_varargs = varargs
            _stypy_temp_lambda_144.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_144', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_144', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to dot(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'y' (line 81)
            y_235925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 52), 'y', False)
            # Processing the call keyword arguments (line 81)
            kwargs_235926 = {}
            # Getting the type of 'H' (line 81)
            H_235923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 46), 'H', False)
            # Obtaining the member 'dot' of a type (line 81)
            dot_235924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 46), H_235923, 'dot')
            # Calling dot(args, kwargs) (line 81)
            dot_call_result_235927 = invoke(stypy.reporting.localization.Localization(__file__, 81, 46), dot_235924, *[y_235925], **kwargs_235926)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'stypy_return_type', dot_call_result_235927)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_144' in the type store
            # Getting the type of 'stypy_return_type' (line 81)
            stypy_return_type_235928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235928)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_144'
            return stypy_return_type_235928

        # Assigning a type to the variable '_stypy_temp_lambda_144' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), '_stypy_temp_lambda_144', _stypy_temp_lambda_144)
        # Getting the type of '_stypy_temp_lambda_144' (line 81)
        _stypy_temp_lambda_144_235929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), '_stypy_temp_lambda_144')
        keyword_235930 = _stypy_temp_lambda_144_235929
        kwargs_235931 = {'fun': keyword_235914, 'x': keyword_235910, 'hess': keyword_235922, 'jac': keyword_235918, 'hessp': keyword_235930}
        # Getting the type of 'KrylovQP' (line 77)
        KrylovQP_235908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'KrylovQP', False)
        # Calling KrylovQP(args, kwargs) (line 77)
        KrylovQP_call_result_235932 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), KrylovQP_235908, *[], **kwargs_235931)
        
        # Assigning a type to the variable 'subprob' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'subprob', KrylovQP_call_result_235932)
        
        # Assigning a Call to a Tuple (line 82):
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_235933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        
        # Call to solve(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'trust_radius' (line 82)
        trust_radius_235936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 82)
        kwargs_235937 = {}
        # Getting the type of 'subprob' (line 82)
        subprob_235934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 82)
        solve_235935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), subprob_235934, 'solve')
        # Calling solve(args, kwargs) (line 82)
        solve_call_result_235938 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), solve_235935, *[trust_radius_235936], **kwargs_235937)
        
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___235939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), solve_call_result_235938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_235940 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___235939, int_235933)
        
        # Assigning a type to the variable 'tuple_var_assignment_235662' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_235662', subscript_call_result_235940)
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_235941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        
        # Call to solve(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'trust_radius' (line 82)
        trust_radius_235944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 82)
        kwargs_235945 = {}
        # Getting the type of 'subprob' (line 82)
        subprob_235942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 82)
        solve_235943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), subprob_235942, 'solve')
        # Calling solve(args, kwargs) (line 82)
        solve_call_result_235946 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), solve_235943, *[trust_radius_235944], **kwargs_235945)
        
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___235947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), solve_call_result_235946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_235948 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___235947, int_235941)
        
        # Assigning a type to the variable 'tuple_var_assignment_235663' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_235663', subscript_call_result_235948)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_235662' (line 82)
        tuple_var_assignment_235662_235949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_235662')
        # Assigning a type to the variable 'p' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'p', tuple_var_assignment_235662_235949)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_235663' (line 82)
        tuple_var_assignment_235663_235950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_235663')
        # Assigning a type to the variable 'hits_boundary' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'hits_boundary', tuple_var_assignment_235663_235950)
        
        # Call to assert_array_almost_equal(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'p' (line 84)
        p_235952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), 'p', False)
        
        # Call to array(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_235955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        float_235956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 46), list_235955, float_235956)
        # Adding element type (line 84)
        float_235957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 46), list_235955, float_235957)
        # Adding element type (line 84)
        float_235958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 46), list_235955, float_235958)
        
        # Processing the call keyword arguments (line 84)
        kwargs_235959 = {}
        # Getting the type of 'np' (line 84)
        np_235953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 37), 'np', False)
        # Obtaining the member 'array' of a type (line 84)
        array_235954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 37), np_235953, 'array')
        # Calling array(args, kwargs) (line 84)
        array_call_result_235960 = invoke(stypy.reporting.localization.Localization(__file__, 84, 37), array_235954, *[list_235955], **kwargs_235959)
        
        # Processing the call keyword arguments (line 84)
        kwargs_235961 = {}
        # Getting the type of 'assert_array_almost_equal' (line 84)
        assert_array_almost_equal_235951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 84)
        assert_array_almost_equal_call_result_235962 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_array_almost_equal_235951, *[p_235952, array_call_result_235960], **kwargs_235961)
        
        
        # Call to assert_almost_equal(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to norm(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to dot(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'p' (line 87)
        p_235969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'p', False)
        # Processing the call keyword arguments (line 87)
        kwargs_235970 = {}
        # Getting the type of 'H' (line 87)
        H_235967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 31), 'H', False)
        # Obtaining the member 'dot' of a type (line 87)
        dot_235968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 31), H_235967, 'dot')
        # Calling dot(args, kwargs) (line 87)
        dot_call_result_235971 = invoke(stypy.reporting.localization.Localization(__file__, 87, 31), dot_235968, *[p_235969], **kwargs_235970)
        
        # Getting the type of 'subprob' (line 87)
        subprob_235972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'subprob', False)
        # Obtaining the member 'lam' of a type (line 87)
        lam_235973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 42), subprob_235972, 'lam')
        # Getting the type of 'p' (line 87)
        p_235974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'p', False)
        # Applying the binary operator '*' (line 87)
        result_mul_235975 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 42), '*', lam_235973, p_235974)
        
        # Applying the binary operator '+' (line 87)
        result_add_235976 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 31), '+', dot_call_result_235971, result_mul_235975)
        
        # Getting the type of 'g' (line 87)
        g_235977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 60), 'g', False)
        # Applying the binary operator '+' (line 87)
        result_add_235978 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 58), '+', result_add_235976, g_235977)
        
        # Processing the call keyword arguments (line 87)
        kwargs_235979 = {}
        # Getting the type of 'np' (line 87)
        np_235964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 87)
        linalg_235965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), np_235964, 'linalg')
        # Obtaining the member 'norm' of a type (line 87)
        norm_235966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), linalg_235965, 'norm')
        # Calling norm(args, kwargs) (line 87)
        norm_call_result_235980 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), norm_235966, *[result_add_235978], **kwargs_235979)
        
        float_235981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 16), 'float')
        # Processing the call keyword arguments (line 86)
        kwargs_235982 = {}
        # Getting the type of 'assert_almost_equal' (line 86)
        assert_almost_equal_235963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 86)
        assert_almost_equal_call_result_235983 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), assert_almost_equal_235963, *[norm_call_result_235980, float_235981], **kwargs_235982)
        
        
        # Call to assert_almost_equal(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to norm(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'p' (line 90)
        p_235988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 43), 'p', False)
        # Processing the call keyword arguments (line 90)
        kwargs_235989 = {}
        # Getting the type of 'np' (line 90)
        np_235985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'np', False)
        # Obtaining the member 'linalg' of a type (line 90)
        linalg_235986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), np_235985, 'linalg')
        # Obtaining the member 'norm' of a type (line 90)
        norm_235987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), linalg_235986, 'norm')
        # Calling norm(args, kwargs) (line 90)
        norm_call_result_235990 = invoke(stypy.reporting.localization.Localization(__file__, 90, 28), norm_235987, *[p_235988], **kwargs_235989)
        
        # Getting the type of 'trust_radius' (line 90)
        trust_radius_235991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 47), 'trust_radius', False)
        # Processing the call keyword arguments (line 90)
        kwargs_235992 = {}
        # Getting the type of 'assert_almost_equal' (line 90)
        assert_almost_equal_235984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 90)
        assert_almost_equal_call_result_235993 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_almost_equal_235984, *[norm_call_result_235990, trust_radius_235991], **kwargs_235992)
        
        
        # Assigning a Num to a Name (line 92):
        
        # Assigning a Num to a Name (line 92):
        float_235994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'float')
        # Assigning a type to the variable 'trust_radius' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'trust_radius', float_235994)
        
        # Assigning a Call to a Tuple (line 93):
        
        # Assigning a Subscript to a Name (line 93):
        
        # Obtaining the type of the subscript
        int_235995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        
        # Call to solve(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'trust_radius' (line 93)
        trust_radius_235998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 93)
        kwargs_235999 = {}
        # Getting the type of 'subprob' (line 93)
        subprob_235996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 93)
        solve_235997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 27), subprob_235996, 'solve')
        # Calling solve(args, kwargs) (line 93)
        solve_call_result_236000 = invoke(stypy.reporting.localization.Localization(__file__, 93, 27), solve_235997, *[trust_radius_235998], **kwargs_235999)
        
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___236001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), solve_call_result_236000, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_236002 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), getitem___236001, int_235995)
        
        # Assigning a type to the variable 'tuple_var_assignment_235664' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_235664', subscript_call_result_236002)
        
        # Assigning a Subscript to a Name (line 93):
        
        # Obtaining the type of the subscript
        int_236003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        
        # Call to solve(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'trust_radius' (line 93)
        trust_radius_236006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 93)
        kwargs_236007 = {}
        # Getting the type of 'subprob' (line 93)
        subprob_236004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 93)
        solve_236005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 27), subprob_236004, 'solve')
        # Calling solve(args, kwargs) (line 93)
        solve_call_result_236008 = invoke(stypy.reporting.localization.Localization(__file__, 93, 27), solve_236005, *[trust_radius_236006], **kwargs_236007)
        
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___236009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), solve_call_result_236008, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_236010 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), getitem___236009, int_236003)
        
        # Assigning a type to the variable 'tuple_var_assignment_235665' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_235665', subscript_call_result_236010)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'tuple_var_assignment_235664' (line 93)
        tuple_var_assignment_235664_236011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_235664')
        # Assigning a type to the variable 'p' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'p', tuple_var_assignment_235664_236011)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'tuple_var_assignment_235665' (line 93)
        tuple_var_assignment_235665_236012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_235665')
        # Assigning a type to the variable 'hits_boundary' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'hits_boundary', tuple_var_assignment_235665_236012)
        
        # Call to assert_array_almost_equal(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'p' (line 95)
        p_236014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 'p', False)
        
        # Call to array(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_236017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        float_236018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 46), list_236017, float_236018)
        # Adding element type (line 95)
        float_236019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 46), list_236017, float_236019)
        # Adding element type (line 95)
        float_236020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 46), list_236017, float_236020)
        
        # Processing the call keyword arguments (line 95)
        kwargs_236021 = {}
        # Getting the type of 'np' (line 95)
        np_236015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 37), 'np', False)
        # Obtaining the member 'array' of a type (line 95)
        array_236016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 37), np_236015, 'array')
        # Calling array(args, kwargs) (line 95)
        array_call_result_236022 = invoke(stypy.reporting.localization.Localization(__file__, 95, 37), array_236016, *[list_236017], **kwargs_236021)
        
        # Processing the call keyword arguments (line 95)
        kwargs_236023 = {}
        # Getting the type of 'assert_array_almost_equal' (line 95)
        assert_array_almost_equal_236013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 95)
        assert_array_almost_equal_call_result_236024 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assert_array_almost_equal_236013, *[p_236014, array_call_result_236022], **kwargs_236023)
        
        
        # Call to assert_almost_equal(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to norm(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to dot(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'p' (line 98)
        p_236031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'p', False)
        # Processing the call keyword arguments (line 98)
        kwargs_236032 = {}
        # Getting the type of 'H' (line 98)
        H_236029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'H', False)
        # Obtaining the member 'dot' of a type (line 98)
        dot_236030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 31), H_236029, 'dot')
        # Calling dot(args, kwargs) (line 98)
        dot_call_result_236033 = invoke(stypy.reporting.localization.Localization(__file__, 98, 31), dot_236030, *[p_236031], **kwargs_236032)
        
        # Getting the type of 'subprob' (line 98)
        subprob_236034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'subprob', False)
        # Obtaining the member 'lam' of a type (line 98)
        lam_236035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 42), subprob_236034, 'lam')
        # Getting the type of 'p' (line 98)
        p_236036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 56), 'p', False)
        # Applying the binary operator '*' (line 98)
        result_mul_236037 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 42), '*', lam_236035, p_236036)
        
        # Applying the binary operator '+' (line 98)
        result_add_236038 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 31), '+', dot_call_result_236033, result_mul_236037)
        
        # Getting the type of 'g' (line 98)
        g_236039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 60), 'g', False)
        # Applying the binary operator '+' (line 98)
        result_add_236040 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 58), '+', result_add_236038, g_236039)
        
        # Processing the call keyword arguments (line 98)
        kwargs_236041 = {}
        # Getting the type of 'np' (line 98)
        np_236026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 98)
        linalg_236027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), np_236026, 'linalg')
        # Obtaining the member 'norm' of a type (line 98)
        norm_236028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), linalg_236027, 'norm')
        # Calling norm(args, kwargs) (line 98)
        norm_call_result_236042 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), norm_236028, *[result_add_236040], **kwargs_236041)
        
        float_236043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'float')
        # Processing the call keyword arguments (line 97)
        kwargs_236044 = {}
        # Getting the type of 'assert_almost_equal' (line 97)
        assert_almost_equal_236025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 97)
        assert_almost_equal_call_result_236045 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_almost_equal_236025, *[norm_call_result_236042, float_236043], **kwargs_236044)
        
        
        # Call to assert_almost_equal(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to norm(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'p' (line 101)
        p_236050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 43), 'p', False)
        # Processing the call keyword arguments (line 101)
        kwargs_236051 = {}
        # Getting the type of 'np' (line 101)
        np_236047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'np', False)
        # Obtaining the member 'linalg' of a type (line 101)
        linalg_236048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), np_236047, 'linalg')
        # Obtaining the member 'norm' of a type (line 101)
        norm_236049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), linalg_236048, 'norm')
        # Calling norm(args, kwargs) (line 101)
        norm_call_result_236052 = invoke(stypy.reporting.localization.Localization(__file__, 101, 28), norm_236049, *[p_236050], **kwargs_236051)
        
        # Getting the type of 'trust_radius' (line 101)
        trust_radius_236053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 47), 'trust_radius', False)
        # Processing the call keyword arguments (line 101)
        kwargs_236054 = {}
        # Getting the type of 'assert_almost_equal' (line 101)
        assert_almost_equal_236046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 101)
        assert_almost_equal_call_result_236055 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assert_almost_equal_236046, *[norm_call_result_236052, trust_radius_236053], **kwargs_236054)
        
        
        # ################# End of 'test_for_the_hard_case(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_the_hard_case' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_236056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_the_hard_case'
        return stypy_return_type_236056


    @norecursion
    def test_for_interior_convergence(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_interior_convergence'
        module_type_store = module_type_store.open_function_context('test_for_interior_convergence', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_localization', localization)
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_function_name', 'TestKrylovQuadraticSubproblem.test_for_interior_convergence')
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_param_names_list', [])
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKrylovQuadraticSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKrylovQuadraticSubproblem.test_for_interior_convergence', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_interior_convergence', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_interior_convergence(...)' code ##################

        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to array(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_236059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_236060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        float_236061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), list_236060, float_236061)
        # Adding element type (line 105)
        float_236062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), list_236060, float_236062)
        # Adding element type (line 105)
        float_236063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), list_236060, float_236063)
        # Adding element type (line 105)
        float_236064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), list_236060, float_236064)
        # Adding element type (line 105)
        float_236065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), list_236060, float_236065)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 21), list_236059, list_236060)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_236066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        float_236067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 22), list_236066, float_236067)
        # Adding element type (line 106)
        float_236068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 22), list_236066, float_236068)
        # Adding element type (line 106)
        float_236069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 22), list_236066, float_236069)
        # Adding element type (line 106)
        float_236070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 22), list_236066, float_236070)
        # Adding element type (line 106)
        float_236071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 72), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 22), list_236066, float_236071)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 21), list_236059, list_236066)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_236072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        float_236073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), list_236072, float_236073)
        # Adding element type (line 107)
        float_236074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), list_236072, float_236074)
        # Adding element type (line 107)
        float_236075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), list_236072, float_236075)
        # Adding element type (line 107)
        float_236076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), list_236072, float_236076)
        # Adding element type (line 107)
        float_236077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 71), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), list_236072, float_236077)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 21), list_236059, list_236072)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_236078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        float_236079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_236078, float_236079)
        # Adding element type (line 108)
        float_236080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_236078, float_236080)
        # Adding element type (line 108)
        float_236081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_236078, float_236081)
        # Adding element type (line 108)
        float_236082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_236078, float_236082)
        # Adding element type (line 108)
        float_236083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 22), list_236078, float_236083)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 21), list_236059, list_236078)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_236084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        float_236085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_236084, float_236085)
        # Adding element type (line 109)
        float_236086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_236084, float_236086)
        # Adding element type (line 109)
        float_236087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_236084, float_236087)
        # Adding element type (line 109)
        float_236088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_236084, float_236088)
        # Adding element type (line 109)
        float_236089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 72), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), list_236084, float_236089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 21), list_236059, list_236084)
        
        # Processing the call keyword arguments (line 105)
        kwargs_236090 = {}
        # Getting the type of 'np' (line 105)
        np_236057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 105)
        array_236058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), np_236057, 'array')
        # Calling array(args, kwargs) (line 105)
        array_call_result_236091 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), array_236058, *[list_236059], **kwargs_236090)
        
        # Assigning a type to the variable 'H' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'H', array_call_result_236091)
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to array(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_236094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        float_236095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_236094, float_236095)
        # Adding element type (line 110)
        float_236096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_236094, float_236096)
        # Adding element type (line 110)
        float_236097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_236094, float_236097)
        # Adding element type (line 110)
        float_236098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_236094, float_236098)
        # Adding element type (line 110)
        float_236099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_236094, float_236099)
        
        # Processing the call keyword arguments (line 110)
        kwargs_236100 = {}
        # Getting the type of 'np' (line 110)
        np_236092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 110)
        array_236093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), np_236092, 'array')
        # Calling array(args, kwargs) (line 110)
        array_call_result_236101 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), array_236093, *[list_236094], **kwargs_236100)
        
        # Assigning a type to the variable 'g' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'g', array_call_result_236101)
        
        # Assigning a Num to a Name (line 111):
        
        # Assigning a Num to a Name (line 111):
        float_236102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 23), 'float')
        # Assigning a type to the variable 'trust_radius' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'trust_radius', float_236102)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to KrylovQP(...): (line 114)
        # Processing the call keyword arguments (line 114)
        int_236104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'int')
        keyword_236105 = int_236104

        @norecursion
        def _stypy_temp_lambda_145(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_145'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_145', 115, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_145.stypy_localization = localization
            _stypy_temp_lambda_145.stypy_type_of_self = None
            _stypy_temp_lambda_145.stypy_type_store = module_type_store
            _stypy_temp_lambda_145.stypy_function_name = '_stypy_temp_lambda_145'
            _stypy_temp_lambda_145.stypy_param_names_list = ['x']
            _stypy_temp_lambda_145.stypy_varargs_param_name = None
            _stypy_temp_lambda_145.stypy_kwargs_param_name = None
            _stypy_temp_lambda_145.stypy_call_defaults = defaults
            _stypy_temp_lambda_145.stypy_call_varargs = varargs
            _stypy_temp_lambda_145.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_145', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_145', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_236106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 41), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), 'stypy_return_type', int_236106)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_145' in the type store
            # Getting the type of 'stypy_return_type' (line 115)
            stypy_return_type_236107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236107)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_145'
            return stypy_return_type_236107

        # Assigning a type to the variable '_stypy_temp_lambda_145' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), '_stypy_temp_lambda_145', _stypy_temp_lambda_145)
        # Getting the type of '_stypy_temp_lambda_145' (line 115)
        _stypy_temp_lambda_145_236108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), '_stypy_temp_lambda_145')
        keyword_236109 = _stypy_temp_lambda_145_236108

        @norecursion
        def _stypy_temp_lambda_146(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_146'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_146', 116, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_146.stypy_localization = localization
            _stypy_temp_lambda_146.stypy_type_of_self = None
            _stypy_temp_lambda_146.stypy_type_store = module_type_store
            _stypy_temp_lambda_146.stypy_function_name = '_stypy_temp_lambda_146'
            _stypy_temp_lambda_146.stypy_param_names_list = ['x']
            _stypy_temp_lambda_146.stypy_varargs_param_name = None
            _stypy_temp_lambda_146.stypy_kwargs_param_name = None
            _stypy_temp_lambda_146.stypy_call_defaults = defaults
            _stypy_temp_lambda_146.stypy_call_varargs = varargs
            _stypy_temp_lambda_146.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_146', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_146', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'g' (line 116)
            g_236110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 41), 'g', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'stypy_return_type', g_236110)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_146' in the type store
            # Getting the type of 'stypy_return_type' (line 116)
            stypy_return_type_236111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236111)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_146'
            return stypy_return_type_236111

        # Assigning a type to the variable '_stypy_temp_lambda_146' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), '_stypy_temp_lambda_146', _stypy_temp_lambda_146)
        # Getting the type of '_stypy_temp_lambda_146' (line 116)
        _stypy_temp_lambda_146_236112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), '_stypy_temp_lambda_146')
        keyword_236113 = _stypy_temp_lambda_146_236112

        @norecursion
        def _stypy_temp_lambda_147(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_147'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_147', 117, 32, True)
            # Passed parameters checking function
            _stypy_temp_lambda_147.stypy_localization = localization
            _stypy_temp_lambda_147.stypy_type_of_self = None
            _stypy_temp_lambda_147.stypy_type_store = module_type_store
            _stypy_temp_lambda_147.stypy_function_name = '_stypy_temp_lambda_147'
            _stypy_temp_lambda_147.stypy_param_names_list = ['x']
            _stypy_temp_lambda_147.stypy_varargs_param_name = None
            _stypy_temp_lambda_147.stypy_kwargs_param_name = None
            _stypy_temp_lambda_147.stypy_call_defaults = defaults
            _stypy_temp_lambda_147.stypy_call_varargs = varargs
            _stypy_temp_lambda_147.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_147', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_147', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 117)
            None_236114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 42), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'stypy_return_type', None_236114)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_147' in the type store
            # Getting the type of 'stypy_return_type' (line 117)
            stypy_return_type_236115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236115)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_147'
            return stypy_return_type_236115

        # Assigning a type to the variable '_stypy_temp_lambda_147' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), '_stypy_temp_lambda_147', _stypy_temp_lambda_147)
        # Getting the type of '_stypy_temp_lambda_147' (line 117)
        _stypy_temp_lambda_147_236116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), '_stypy_temp_lambda_147')
        keyword_236117 = _stypy_temp_lambda_147_236116

        @norecursion
        def _stypy_temp_lambda_148(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_148'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_148', 118, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_148.stypy_localization = localization
            _stypy_temp_lambda_148.stypy_type_of_self = None
            _stypy_temp_lambda_148.stypy_type_store = module_type_store
            _stypy_temp_lambda_148.stypy_function_name = '_stypy_temp_lambda_148'
            _stypy_temp_lambda_148.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_148.stypy_varargs_param_name = None
            _stypy_temp_lambda_148.stypy_kwargs_param_name = None
            _stypy_temp_lambda_148.stypy_call_defaults = defaults
            _stypy_temp_lambda_148.stypy_call_varargs = varargs
            _stypy_temp_lambda_148.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_148', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_148', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to dot(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'y' (line 118)
            y_236120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 52), 'y', False)
            # Processing the call keyword arguments (line 118)
            kwargs_236121 = {}
            # Getting the type of 'H' (line 118)
            H_236118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 46), 'H', False)
            # Obtaining the member 'dot' of a type (line 118)
            dot_236119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 46), H_236118, 'dot')
            # Calling dot(args, kwargs) (line 118)
            dot_call_result_236122 = invoke(stypy.reporting.localization.Localization(__file__, 118, 46), dot_236119, *[y_236120], **kwargs_236121)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'stypy_return_type', dot_call_result_236122)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_148' in the type store
            # Getting the type of 'stypy_return_type' (line 118)
            stypy_return_type_236123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236123)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_148'
            return stypy_return_type_236123

        # Assigning a type to the variable '_stypy_temp_lambda_148' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), '_stypy_temp_lambda_148', _stypy_temp_lambda_148)
        # Getting the type of '_stypy_temp_lambda_148' (line 118)
        _stypy_temp_lambda_148_236124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), '_stypy_temp_lambda_148')
        keyword_236125 = _stypy_temp_lambda_148_236124
        kwargs_236126 = {'fun': keyword_236109, 'x': keyword_236105, 'hess': keyword_236117, 'jac': keyword_236113, 'hessp': keyword_236125}
        # Getting the type of 'KrylovQP' (line 114)
        KrylovQP_236103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'KrylovQP', False)
        # Calling KrylovQP(args, kwargs) (line 114)
        KrylovQP_call_result_236127 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), KrylovQP_236103, *[], **kwargs_236126)
        
        # Assigning a type to the variable 'subprob' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'subprob', KrylovQP_call_result_236127)
        
        # Assigning a Call to a Tuple (line 119):
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_236128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'int')
        
        # Call to solve(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'trust_radius' (line 119)
        trust_radius_236131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 119)
        kwargs_236132 = {}
        # Getting the type of 'subprob' (line 119)
        subprob_236129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 119)
        solve_236130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 27), subprob_236129, 'solve')
        # Calling solve(args, kwargs) (line 119)
        solve_call_result_236133 = invoke(stypy.reporting.localization.Localization(__file__, 119, 27), solve_236130, *[trust_radius_236131], **kwargs_236132)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___236134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), solve_call_result_236133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_236135 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), getitem___236134, int_236128)
        
        # Assigning a type to the variable 'tuple_var_assignment_235666' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_235666', subscript_call_result_236135)
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_236136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'int')
        
        # Call to solve(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'trust_radius' (line 119)
        trust_radius_236139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 119)
        kwargs_236140 = {}
        # Getting the type of 'subprob' (line 119)
        subprob_236137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 119)
        solve_236138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 27), subprob_236137, 'solve')
        # Calling solve(args, kwargs) (line 119)
        solve_call_result_236141 = invoke(stypy.reporting.localization.Localization(__file__, 119, 27), solve_236138, *[trust_radius_236139], **kwargs_236140)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___236142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), solve_call_result_236141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_236143 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), getitem___236142, int_236136)
        
        # Assigning a type to the variable 'tuple_var_assignment_235667' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_235667', subscript_call_result_236143)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'tuple_var_assignment_235666' (line 119)
        tuple_var_assignment_235666_236144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_235666')
        # Assigning a type to the variable 'p' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'p', tuple_var_assignment_235666_236144)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'tuple_var_assignment_235667' (line 119)
        tuple_var_assignment_235667_236145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_235667')
        # Assigning a type to the variable 'hits_boundary' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'hits_boundary', tuple_var_assignment_235667_236145)
        
        # Call to assert_almost_equal(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to norm(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to dot(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'p' (line 123)
        p_236152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'p', False)
        # Processing the call keyword arguments (line 123)
        kwargs_236153 = {}
        # Getting the type of 'H' (line 123)
        H_236150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'H', False)
        # Obtaining the member 'dot' of a type (line 123)
        dot_236151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 31), H_236150, 'dot')
        # Calling dot(args, kwargs) (line 123)
        dot_call_result_236154 = invoke(stypy.reporting.localization.Localization(__file__, 123, 31), dot_236151, *[p_236152], **kwargs_236153)
        
        # Getting the type of 'subprob' (line 123)
        subprob_236155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 42), 'subprob', False)
        # Obtaining the member 'lam' of a type (line 123)
        lam_236156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 42), subprob_236155, 'lam')
        # Getting the type of 'p' (line 123)
        p_236157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 56), 'p', False)
        # Applying the binary operator '*' (line 123)
        result_mul_236158 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 42), '*', lam_236156, p_236157)
        
        # Applying the binary operator '+' (line 123)
        result_add_236159 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 31), '+', dot_call_result_236154, result_mul_236158)
        
        # Getting the type of 'g' (line 123)
        g_236160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 60), 'g', False)
        # Applying the binary operator '+' (line 123)
        result_add_236161 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 58), '+', result_add_236159, g_236160)
        
        # Processing the call keyword arguments (line 123)
        kwargs_236162 = {}
        # Getting the type of 'np' (line 123)
        np_236147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 123)
        linalg_236148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), np_236147, 'linalg')
        # Obtaining the member 'norm' of a type (line 123)
        norm_236149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), linalg_236148, 'norm')
        # Calling norm(args, kwargs) (line 123)
        norm_call_result_236163 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), norm_236149, *[result_add_236161], **kwargs_236162)
        
        float_236164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 16), 'float')
        # Processing the call keyword arguments (line 122)
        kwargs_236165 = {}
        # Getting the type of 'assert_almost_equal' (line 122)
        assert_almost_equal_236146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 122)
        assert_almost_equal_call_result_236166 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assert_almost_equal_236146, *[norm_call_result_236163, float_236164], **kwargs_236165)
        
        
        # Call to assert_array_almost_equal(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'p' (line 126)
        p_236168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), 'p', False)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_236169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        float_236170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 37), list_236169, float_236170)
        # Adding element type (line 126)
        float_236171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 37), list_236169, float_236171)
        # Adding element type (line 126)
        float_236172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 37), list_236169, float_236172)
        # Adding element type (line 126)
        float_236173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 37), list_236169, float_236173)
        # Adding element type (line 126)
        float_236174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 37), list_236169, float_236174)
        
        # Processing the call keyword arguments (line 126)
        kwargs_236175 = {}
        # Getting the type of 'assert_array_almost_equal' (line 126)
        assert_array_almost_equal_236167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 126)
        assert_array_almost_equal_call_result_236176 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assert_array_almost_equal_236167, *[p_236168, list_236169], **kwargs_236175)
        
        
        # Call to assert_array_almost_equal(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'hits_boundary' (line 128)
        hits_boundary_236178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'hits_boundary', False)
        # Getting the type of 'False' (line 128)
        False_236179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 49), 'False', False)
        # Processing the call keyword arguments (line 128)
        kwargs_236180 = {}
        # Getting the type of 'assert_array_almost_equal' (line 128)
        assert_array_almost_equal_236177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 128)
        assert_array_almost_equal_call_result_236181 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), assert_array_almost_equal_236177, *[hits_boundary_236178, False_236179], **kwargs_236180)
        
        
        # ################# End of 'test_for_interior_convergence(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_interior_convergence' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_236182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236182)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_interior_convergence'
        return stypy_return_type_236182


    @norecursion
    def test_for_very_close_to_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_very_close_to_zero'
        module_type_store = module_type_store.open_function_context('test_for_very_close_to_zero', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_localization', localization)
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_function_name', 'TestKrylovQuadraticSubproblem.test_for_very_close_to_zero')
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKrylovQuadraticSubproblem.test_for_very_close_to_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKrylovQuadraticSubproblem.test_for_very_close_to_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_very_close_to_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_very_close_to_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 132):
        
        # Assigning a Call to a Name (line 132):
        
        # Call to array(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_236185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_236186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        float_236187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 22), list_236186, float_236187)
        # Adding element type (line 132)
        float_236188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 22), list_236186, float_236188)
        # Adding element type (line 132)
        float_236189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 22), list_236186, float_236189)
        # Adding element type (line 132)
        float_236190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 22), list_236186, float_236190)
        # Adding element type (line 132)
        float_236191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 72), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 22), list_236186, float_236191)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_236185, list_236186)
        # Adding element type (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_236192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        float_236193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_236192, float_236193)
        # Adding element type (line 133)
        float_236194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_236192, float_236194)
        # Adding element type (line 133)
        float_236195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_236192, float_236195)
        # Adding element type (line 133)
        float_236196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_236192, float_236196)
        # Adding element type (line 133)
        float_236197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_236192, float_236197)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_236185, list_236192)
        # Adding element type (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_236198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        float_236199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_236198, float_236199)
        # Adding element type (line 134)
        float_236200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_236198, float_236200)
        # Adding element type (line 134)
        float_236201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_236198, float_236201)
        # Adding element type (line 134)
        float_236202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_236198, float_236202)
        # Adding element type (line 134)
        float_236203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_236198, float_236203)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_236185, list_236198)
        # Adding element type (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_236204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        float_236205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 22), list_236204, float_236205)
        # Adding element type (line 135)
        float_236206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 22), list_236204, float_236206)
        # Adding element type (line 135)
        float_236207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 22), list_236204, float_236207)
        # Adding element type (line 135)
        float_236208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 22), list_236204, float_236208)
        # Adding element type (line 135)
        float_236209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 22), list_236204, float_236209)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_236185, list_236204)
        # Adding element type (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_236210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        float_236211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 22), list_236210, float_236211)
        # Adding element type (line 136)
        float_236212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 22), list_236210, float_236212)
        # Adding element type (line 136)
        float_236213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 22), list_236210, float_236213)
        # Adding element type (line 136)
        float_236214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 22), list_236210, float_236214)
        # Adding element type (line 136)
        float_236215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 74), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 22), list_236210, float_236215)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_236185, list_236210)
        
        # Processing the call keyword arguments (line 132)
        kwargs_236216 = {}
        # Getting the type of 'np' (line 132)
        np_236183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 132)
        array_236184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), np_236183, 'array')
        # Calling array(args, kwargs) (line 132)
        array_call_result_236217 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), array_236184, *[list_236185], **kwargs_236216)
        
        # Assigning a type to the variable 'H' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'H', array_call_result_236217)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to array(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_236220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        int_236221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), list_236220, int_236221)
        # Adding element type (line 137)
        int_236222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), list_236220, int_236222)
        # Adding element type (line 137)
        int_236223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), list_236220, int_236223)
        # Adding element type (line 137)
        int_236224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), list_236220, int_236224)
        # Adding element type (line 137)
        float_236225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), list_236220, float_236225)
        
        # Processing the call keyword arguments (line 137)
        kwargs_236226 = {}
        # Getting the type of 'np' (line 137)
        np_236218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 137)
        array_236219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), np_236218, 'array')
        # Calling array(args, kwargs) (line 137)
        array_call_result_236227 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), array_236219, *[list_236220], **kwargs_236226)
        
        # Assigning a type to the variable 'g' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'g', array_call_result_236227)
        
        # Assigning a Num to a Name (line 138):
        
        # Assigning a Num to a Name (line 138):
        float_236228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'float')
        # Assigning a type to the variable 'trust_radius' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'trust_radius', float_236228)
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to KrylovQP(...): (line 141)
        # Processing the call keyword arguments (line 141)
        int_236230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'int')
        keyword_236231 = int_236230

        @norecursion
        def _stypy_temp_lambda_149(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_149'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_149', 142, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_149.stypy_localization = localization
            _stypy_temp_lambda_149.stypy_type_of_self = None
            _stypy_temp_lambda_149.stypy_type_store = module_type_store
            _stypy_temp_lambda_149.stypy_function_name = '_stypy_temp_lambda_149'
            _stypy_temp_lambda_149.stypy_param_names_list = ['x']
            _stypy_temp_lambda_149.stypy_varargs_param_name = None
            _stypy_temp_lambda_149.stypy_kwargs_param_name = None
            _stypy_temp_lambda_149.stypy_call_defaults = defaults
            _stypy_temp_lambda_149.stypy_call_varargs = varargs
            _stypy_temp_lambda_149.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_149', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_149', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_236232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 41), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'stypy_return_type', int_236232)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_149' in the type store
            # Getting the type of 'stypy_return_type' (line 142)
            stypy_return_type_236233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236233)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_149'
            return stypy_return_type_236233

        # Assigning a type to the variable '_stypy_temp_lambda_149' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), '_stypy_temp_lambda_149', _stypy_temp_lambda_149)
        # Getting the type of '_stypy_temp_lambda_149' (line 142)
        _stypy_temp_lambda_149_236234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), '_stypy_temp_lambda_149')
        keyword_236235 = _stypy_temp_lambda_149_236234

        @norecursion
        def _stypy_temp_lambda_150(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_150'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_150', 143, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_150.stypy_localization = localization
            _stypy_temp_lambda_150.stypy_type_of_self = None
            _stypy_temp_lambda_150.stypy_type_store = module_type_store
            _stypy_temp_lambda_150.stypy_function_name = '_stypy_temp_lambda_150'
            _stypy_temp_lambda_150.stypy_param_names_list = ['x']
            _stypy_temp_lambda_150.stypy_varargs_param_name = None
            _stypy_temp_lambda_150.stypy_kwargs_param_name = None
            _stypy_temp_lambda_150.stypy_call_defaults = defaults
            _stypy_temp_lambda_150.stypy_call_varargs = varargs
            _stypy_temp_lambda_150.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_150', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_150', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'g' (line 143)
            g_236236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'g', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'stypy_return_type', g_236236)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_150' in the type store
            # Getting the type of 'stypy_return_type' (line 143)
            stypy_return_type_236237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236237)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_150'
            return stypy_return_type_236237

        # Assigning a type to the variable '_stypy_temp_lambda_150' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), '_stypy_temp_lambda_150', _stypy_temp_lambda_150)
        # Getting the type of '_stypy_temp_lambda_150' (line 143)
        _stypy_temp_lambda_150_236238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), '_stypy_temp_lambda_150')
        keyword_236239 = _stypy_temp_lambda_150_236238

        @norecursion
        def _stypy_temp_lambda_151(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_151'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_151', 144, 32, True)
            # Passed parameters checking function
            _stypy_temp_lambda_151.stypy_localization = localization
            _stypy_temp_lambda_151.stypy_type_of_self = None
            _stypy_temp_lambda_151.stypy_type_store = module_type_store
            _stypy_temp_lambda_151.stypy_function_name = '_stypy_temp_lambda_151'
            _stypy_temp_lambda_151.stypy_param_names_list = ['x']
            _stypy_temp_lambda_151.stypy_varargs_param_name = None
            _stypy_temp_lambda_151.stypy_kwargs_param_name = None
            _stypy_temp_lambda_151.stypy_call_defaults = defaults
            _stypy_temp_lambda_151.stypy_call_varargs = varargs
            _stypy_temp_lambda_151.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_151', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_151', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 144)
            None_236240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 42), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'stypy_return_type', None_236240)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_151' in the type store
            # Getting the type of 'stypy_return_type' (line 144)
            stypy_return_type_236241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236241)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_151'
            return stypy_return_type_236241

        # Assigning a type to the variable '_stypy_temp_lambda_151' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), '_stypy_temp_lambda_151', _stypy_temp_lambda_151)
        # Getting the type of '_stypy_temp_lambda_151' (line 144)
        _stypy_temp_lambda_151_236242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), '_stypy_temp_lambda_151')
        keyword_236243 = _stypy_temp_lambda_151_236242

        @norecursion
        def _stypy_temp_lambda_152(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_152'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_152', 145, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_152.stypy_localization = localization
            _stypy_temp_lambda_152.stypy_type_of_self = None
            _stypy_temp_lambda_152.stypy_type_store = module_type_store
            _stypy_temp_lambda_152.stypy_function_name = '_stypy_temp_lambda_152'
            _stypy_temp_lambda_152.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_152.stypy_varargs_param_name = None
            _stypy_temp_lambda_152.stypy_kwargs_param_name = None
            _stypy_temp_lambda_152.stypy_call_defaults = defaults
            _stypy_temp_lambda_152.stypy_call_varargs = varargs
            _stypy_temp_lambda_152.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_152', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_152', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to dot(...): (line 145)
            # Processing the call arguments (line 145)
            # Getting the type of 'y' (line 145)
            y_236246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 52), 'y', False)
            # Processing the call keyword arguments (line 145)
            kwargs_236247 = {}
            # Getting the type of 'H' (line 145)
            H_236244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 46), 'H', False)
            # Obtaining the member 'dot' of a type (line 145)
            dot_236245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 46), H_236244, 'dot')
            # Calling dot(args, kwargs) (line 145)
            dot_call_result_236248 = invoke(stypy.reporting.localization.Localization(__file__, 145, 46), dot_236245, *[y_236246], **kwargs_236247)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), 'stypy_return_type', dot_call_result_236248)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_152' in the type store
            # Getting the type of 'stypy_return_type' (line 145)
            stypy_return_type_236249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236249)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_152'
            return stypy_return_type_236249

        # Assigning a type to the variable '_stypy_temp_lambda_152' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), '_stypy_temp_lambda_152', _stypy_temp_lambda_152)
        # Getting the type of '_stypy_temp_lambda_152' (line 145)
        _stypy_temp_lambda_152_236250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), '_stypy_temp_lambda_152')
        keyword_236251 = _stypy_temp_lambda_152_236250
        kwargs_236252 = {'fun': keyword_236235, 'x': keyword_236231, 'hess': keyword_236243, 'jac': keyword_236239, 'hessp': keyword_236251}
        # Getting the type of 'KrylovQP' (line 141)
        KrylovQP_236229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 18), 'KrylovQP', False)
        # Calling KrylovQP(args, kwargs) (line 141)
        KrylovQP_call_result_236253 = invoke(stypy.reporting.localization.Localization(__file__, 141, 18), KrylovQP_236229, *[], **kwargs_236252)
        
        # Assigning a type to the variable 'subprob' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'subprob', KrylovQP_call_result_236253)
        
        # Assigning a Call to a Tuple (line 146):
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_236254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to solve(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'trust_radius' (line 146)
        trust_radius_236257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 146)
        kwargs_236258 = {}
        # Getting the type of 'subprob' (line 146)
        subprob_236255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 146)
        solve_236256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 27), subprob_236255, 'solve')
        # Calling solve(args, kwargs) (line 146)
        solve_call_result_236259 = invoke(stypy.reporting.localization.Localization(__file__, 146, 27), solve_236256, *[trust_radius_236257], **kwargs_236258)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___236260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), solve_call_result_236259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_236261 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___236260, int_236254)
        
        # Assigning a type to the variable 'tuple_var_assignment_235668' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_235668', subscript_call_result_236261)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_236262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to solve(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'trust_radius' (line 146)
        trust_radius_236265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 146)
        kwargs_236266 = {}
        # Getting the type of 'subprob' (line 146)
        subprob_236263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 146)
        solve_236264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 27), subprob_236263, 'solve')
        # Calling solve(args, kwargs) (line 146)
        solve_call_result_236267 = invoke(stypy.reporting.localization.Localization(__file__, 146, 27), solve_236264, *[trust_radius_236265], **kwargs_236266)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___236268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), solve_call_result_236267, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_236269 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___236268, int_236262)
        
        # Assigning a type to the variable 'tuple_var_assignment_235669' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_235669', subscript_call_result_236269)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_235668' (line 146)
        tuple_var_assignment_235668_236270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_235668')
        # Assigning a type to the variable 'p' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'p', tuple_var_assignment_235668_236270)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_235669' (line 146)
        tuple_var_assignment_235669_236271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_235669')
        # Assigning a type to the variable 'hits_boundary' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'hits_boundary', tuple_var_assignment_235669_236271)
        
        # Call to assert_almost_equal(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to norm(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Call to dot(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'p' (line 150)
        p_236278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'p', False)
        # Processing the call keyword arguments (line 150)
        kwargs_236279 = {}
        # Getting the type of 'H' (line 150)
        H_236276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 31), 'H', False)
        # Obtaining the member 'dot' of a type (line 150)
        dot_236277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 31), H_236276, 'dot')
        # Calling dot(args, kwargs) (line 150)
        dot_call_result_236280 = invoke(stypy.reporting.localization.Localization(__file__, 150, 31), dot_236277, *[p_236278], **kwargs_236279)
        
        # Getting the type of 'subprob' (line 150)
        subprob_236281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 42), 'subprob', False)
        # Obtaining the member 'lam' of a type (line 150)
        lam_236282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 42), subprob_236281, 'lam')
        # Getting the type of 'p' (line 150)
        p_236283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 56), 'p', False)
        # Applying the binary operator '*' (line 150)
        result_mul_236284 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 42), '*', lam_236282, p_236283)
        
        # Applying the binary operator '+' (line 150)
        result_add_236285 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 31), '+', dot_call_result_236280, result_mul_236284)
        
        # Getting the type of 'g' (line 150)
        g_236286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 60), 'g', False)
        # Applying the binary operator '+' (line 150)
        result_add_236287 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 58), '+', result_add_236285, g_236286)
        
        # Processing the call keyword arguments (line 150)
        kwargs_236288 = {}
        # Getting the type of 'np' (line 150)
        np_236273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 150)
        linalg_236274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), np_236273, 'linalg')
        # Obtaining the member 'norm' of a type (line 150)
        norm_236275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), linalg_236274, 'norm')
        # Calling norm(args, kwargs) (line 150)
        norm_call_result_236289 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), norm_236275, *[result_add_236287], **kwargs_236288)
        
        float_236290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 16), 'float')
        # Processing the call keyword arguments (line 149)
        kwargs_236291 = {}
        # Getting the type of 'assert_almost_equal' (line 149)
        assert_almost_equal_236272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 149)
        assert_almost_equal_call_result_236292 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assert_almost_equal_236272, *[norm_call_result_236289, float_236290], **kwargs_236291)
        
        
        # Call to assert_almost_equal(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Call to norm(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'p' (line 153)
        p_236297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 43), 'p', False)
        # Processing the call keyword arguments (line 153)
        kwargs_236298 = {}
        # Getting the type of 'np' (line 153)
        np_236294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'np', False)
        # Obtaining the member 'linalg' of a type (line 153)
        linalg_236295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 28), np_236294, 'linalg')
        # Obtaining the member 'norm' of a type (line 153)
        norm_236296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 28), linalg_236295, 'norm')
        # Calling norm(args, kwargs) (line 153)
        norm_call_result_236299 = invoke(stypy.reporting.localization.Localization(__file__, 153, 28), norm_236296, *[p_236297], **kwargs_236298)
        
        # Getting the type of 'trust_radius' (line 153)
        trust_radius_236300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 47), 'trust_radius', False)
        # Processing the call keyword arguments (line 153)
        kwargs_236301 = {}
        # Getting the type of 'assert_almost_equal' (line 153)
        assert_almost_equal_236293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 153)
        assert_almost_equal_call_result_236302 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert_almost_equal_236293, *[norm_call_result_236299, trust_radius_236300], **kwargs_236301)
        
        
        # Call to assert_array_almost_equal(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'p' (line 155)
        p_236304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'p', False)
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_236305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        # Adding element type (line 155)
        float_236306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 37), list_236305, float_236306)
        # Adding element type (line 155)
        float_236307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 37), list_236305, float_236307)
        # Adding element type (line 155)
        float_236308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 37), list_236305, float_236308)
        # Adding element type (line 155)
        float_236309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 37), list_236305, float_236309)
        # Adding element type (line 155)
        float_236310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 37), list_236305, float_236310)
        
        # Processing the call keyword arguments (line 155)
        kwargs_236311 = {}
        # Getting the type of 'assert_array_almost_equal' (line 155)
        assert_array_almost_equal_236303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 155)
        assert_array_almost_equal_call_result_236312 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), assert_array_almost_equal_236303, *[p_236304, list_236305], **kwargs_236311)
        
        
        # Call to assert_array_almost_equal(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'hits_boundary' (line 158)
        hits_boundary_236314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 34), 'hits_boundary', False)
        # Getting the type of 'True' (line 158)
        True_236315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 49), 'True', False)
        # Processing the call keyword arguments (line 158)
        kwargs_236316 = {}
        # Getting the type of 'assert_array_almost_equal' (line 158)
        assert_array_almost_equal_236313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 158)
        assert_array_almost_equal_call_result_236317 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), assert_array_almost_equal_236313, *[hits_boundary_236314, True_236315], **kwargs_236316)
        
        
        # ################# End of 'test_for_very_close_to_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_very_close_to_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_236318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_very_close_to_zero'
        return stypy_return_type_236318


    @norecursion
    def test_disp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_disp'
        module_type_store = module_type_store.open_function_context('test_disp', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_localization', localization)
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_function_name', 'TestKrylovQuadraticSubproblem.test_disp')
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_param_names_list', ['capsys'])
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKrylovQuadraticSubproblem.test_disp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKrylovQuadraticSubproblem.test_disp', ['capsys'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_disp', localization, ['capsys'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_disp(...)' code ##################

        
        # Assigning a UnaryOp to a Name (line 161):
        
        # Assigning a UnaryOp to a Name (line 161):
        
        
        # Call to eye(...): (line 161)
        # Processing the call arguments (line 161)
        int_236321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'int')
        # Processing the call keyword arguments (line 161)
        kwargs_236322 = {}
        # Getting the type of 'np' (line 161)
        np_236319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'np', False)
        # Obtaining the member 'eye' of a type (line 161)
        eye_236320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 13), np_236319, 'eye')
        # Calling eye(args, kwargs) (line 161)
        eye_call_result_236323 = invoke(stypy.reporting.localization.Localization(__file__, 161, 13), eye_236320, *[int_236321], **kwargs_236322)
        
        # Applying the 'usub' unary operator (line 161)
        result___neg___236324 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 12), 'usub', eye_call_result_236323)
        
        # Assigning a type to the variable 'H' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'H', result___neg___236324)
        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to array(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_236327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        int_236328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 21), list_236327, int_236328)
        # Adding element type (line 162)
        int_236329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 21), list_236327, int_236329)
        # Adding element type (line 162)
        int_236330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 21), list_236327, int_236330)
        # Adding element type (line 162)
        int_236331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 21), list_236327, int_236331)
        # Adding element type (line 162)
        float_236332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 21), list_236327, float_236332)
        
        # Processing the call keyword arguments (line 162)
        kwargs_236333 = {}
        # Getting the type of 'np' (line 162)
        np_236325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 162)
        array_236326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), np_236325, 'array')
        # Calling array(args, kwargs) (line 162)
        array_call_result_236334 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), array_236326, *[list_236327], **kwargs_236333)
        
        # Assigning a type to the variable 'g' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'g', array_call_result_236334)
        
        # Assigning a Num to a Name (line 163):
        
        # Assigning a Num to a Name (line 163):
        float_236335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 23), 'float')
        # Assigning a type to the variable 'trust_radius' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'trust_radius', float_236335)
        
        # Assigning a Call to a Name (line 165):
        
        # Assigning a Call to a Name (line 165):
        
        # Call to KrylovQP_disp(...): (line 165)
        # Processing the call keyword arguments (line 165)
        int_236337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 34), 'int')
        keyword_236338 = int_236337

        @norecursion
        def _stypy_temp_lambda_153(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_153'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_153', 166, 36, True)
            # Passed parameters checking function
            _stypy_temp_lambda_153.stypy_localization = localization
            _stypy_temp_lambda_153.stypy_type_of_self = None
            _stypy_temp_lambda_153.stypy_type_store = module_type_store
            _stypy_temp_lambda_153.stypy_function_name = '_stypy_temp_lambda_153'
            _stypy_temp_lambda_153.stypy_param_names_list = ['x']
            _stypy_temp_lambda_153.stypy_varargs_param_name = None
            _stypy_temp_lambda_153.stypy_kwargs_param_name = None
            _stypy_temp_lambda_153.stypy_call_defaults = defaults
            _stypy_temp_lambda_153.stypy_call_varargs = varargs
            _stypy_temp_lambda_153.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_153', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_153', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_236339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 46), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), 'stypy_return_type', int_236339)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_153' in the type store
            # Getting the type of 'stypy_return_type' (line 166)
            stypy_return_type_236340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236340)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_153'
            return stypy_return_type_236340

        # Assigning a type to the variable '_stypy_temp_lambda_153' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), '_stypy_temp_lambda_153', _stypy_temp_lambda_153)
        # Getting the type of '_stypy_temp_lambda_153' (line 166)
        _stypy_temp_lambda_153_236341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), '_stypy_temp_lambda_153')
        keyword_236342 = _stypy_temp_lambda_153_236341

        @norecursion
        def _stypy_temp_lambda_154(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_154'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_154', 167, 36, True)
            # Passed parameters checking function
            _stypy_temp_lambda_154.stypy_localization = localization
            _stypy_temp_lambda_154.stypy_type_of_self = None
            _stypy_temp_lambda_154.stypy_type_store = module_type_store
            _stypy_temp_lambda_154.stypy_function_name = '_stypy_temp_lambda_154'
            _stypy_temp_lambda_154.stypy_param_names_list = ['x']
            _stypy_temp_lambda_154.stypy_varargs_param_name = None
            _stypy_temp_lambda_154.stypy_kwargs_param_name = None
            _stypy_temp_lambda_154.stypy_call_defaults = defaults
            _stypy_temp_lambda_154.stypy_call_varargs = varargs
            _stypy_temp_lambda_154.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_154', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_154', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'g' (line 167)
            g_236343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 46), 'g', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 36), 'stypy_return_type', g_236343)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_154' in the type store
            # Getting the type of 'stypy_return_type' (line 167)
            stypy_return_type_236344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 36), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236344)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_154'
            return stypy_return_type_236344

        # Assigning a type to the variable '_stypy_temp_lambda_154' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 36), '_stypy_temp_lambda_154', _stypy_temp_lambda_154)
        # Getting the type of '_stypy_temp_lambda_154' (line 167)
        _stypy_temp_lambda_154_236345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 36), '_stypy_temp_lambda_154')
        keyword_236346 = _stypy_temp_lambda_154_236345

        @norecursion
        def _stypy_temp_lambda_155(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_155'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_155', 168, 37, True)
            # Passed parameters checking function
            _stypy_temp_lambda_155.stypy_localization = localization
            _stypy_temp_lambda_155.stypy_type_of_self = None
            _stypy_temp_lambda_155.stypy_type_store = module_type_store
            _stypy_temp_lambda_155.stypy_function_name = '_stypy_temp_lambda_155'
            _stypy_temp_lambda_155.stypy_param_names_list = ['x']
            _stypy_temp_lambda_155.stypy_varargs_param_name = None
            _stypy_temp_lambda_155.stypy_kwargs_param_name = None
            _stypy_temp_lambda_155.stypy_call_defaults = defaults
            _stypy_temp_lambda_155.stypy_call_varargs = varargs
            _stypy_temp_lambda_155.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_155', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_155', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 168)
            None_236347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 47), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'stypy_return_type', None_236347)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_155' in the type store
            # Getting the type of 'stypy_return_type' (line 168)
            stypy_return_type_236348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236348)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_155'
            return stypy_return_type_236348

        # Assigning a type to the variable '_stypy_temp_lambda_155' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), '_stypy_temp_lambda_155', _stypy_temp_lambda_155)
        # Getting the type of '_stypy_temp_lambda_155' (line 168)
        _stypy_temp_lambda_155_236349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), '_stypy_temp_lambda_155')
        keyword_236350 = _stypy_temp_lambda_155_236349

        @norecursion
        def _stypy_temp_lambda_156(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_156'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_156', 169, 38, True)
            # Passed parameters checking function
            _stypy_temp_lambda_156.stypy_localization = localization
            _stypy_temp_lambda_156.stypy_type_of_self = None
            _stypy_temp_lambda_156.stypy_type_store = module_type_store
            _stypy_temp_lambda_156.stypy_function_name = '_stypy_temp_lambda_156'
            _stypy_temp_lambda_156.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_156.stypy_varargs_param_name = None
            _stypy_temp_lambda_156.stypy_kwargs_param_name = None
            _stypy_temp_lambda_156.stypy_call_defaults = defaults
            _stypy_temp_lambda_156.stypy_call_varargs = varargs
            _stypy_temp_lambda_156.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_156', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_156', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to dot(...): (line 169)
            # Processing the call arguments (line 169)
            # Getting the type of 'y' (line 169)
            y_236353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 57), 'y', False)
            # Processing the call keyword arguments (line 169)
            kwargs_236354 = {}
            # Getting the type of 'H' (line 169)
            H_236351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 51), 'H', False)
            # Obtaining the member 'dot' of a type (line 169)
            dot_236352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 51), H_236351, 'dot')
            # Calling dot(args, kwargs) (line 169)
            dot_call_result_236355 = invoke(stypy.reporting.localization.Localization(__file__, 169, 51), dot_236352, *[y_236353], **kwargs_236354)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 38), 'stypy_return_type', dot_call_result_236355)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_156' in the type store
            # Getting the type of 'stypy_return_type' (line 169)
            stypy_return_type_236356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 38), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_236356)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_156'
            return stypy_return_type_236356

        # Assigning a type to the variable '_stypy_temp_lambda_156' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 38), '_stypy_temp_lambda_156', _stypy_temp_lambda_156)
        # Getting the type of '_stypy_temp_lambda_156' (line 169)
        _stypy_temp_lambda_156_236357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 38), '_stypy_temp_lambda_156')
        keyword_236358 = _stypy_temp_lambda_156_236357
        kwargs_236359 = {'fun': keyword_236342, 'x': keyword_236338, 'hess': keyword_236350, 'jac': keyword_236346, 'hessp': keyword_236358}
        # Getting the type of 'KrylovQP_disp' (line 165)
        KrylovQP_disp_236336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'KrylovQP_disp', False)
        # Calling KrylovQP_disp(args, kwargs) (line 165)
        KrylovQP_disp_call_result_236360 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), KrylovQP_disp_236336, *[], **kwargs_236359)
        
        # Assigning a type to the variable 'subprob' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'subprob', KrylovQP_disp_call_result_236360)
        
        # Assigning a Call to a Tuple (line 170):
        
        # Assigning a Subscript to a Name (line 170):
        
        # Obtaining the type of the subscript
        int_236361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'int')
        
        # Call to solve(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'trust_radius' (line 170)
        trust_radius_236364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 170)
        kwargs_236365 = {}
        # Getting the type of 'subprob' (line 170)
        subprob_236362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 170)
        solve_236363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 27), subprob_236362, 'solve')
        # Calling solve(args, kwargs) (line 170)
        solve_call_result_236366 = invoke(stypy.reporting.localization.Localization(__file__, 170, 27), solve_236363, *[trust_radius_236364], **kwargs_236365)
        
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___236367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), solve_call_result_236366, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_236368 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), getitem___236367, int_236361)
        
        # Assigning a type to the variable 'tuple_var_assignment_235670' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_235670', subscript_call_result_236368)
        
        # Assigning a Subscript to a Name (line 170):
        
        # Obtaining the type of the subscript
        int_236369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'int')
        
        # Call to solve(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'trust_radius' (line 170)
        trust_radius_236372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 170)
        kwargs_236373 = {}
        # Getting the type of 'subprob' (line 170)
        subprob_236370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 170)
        solve_236371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 27), subprob_236370, 'solve')
        # Calling solve(args, kwargs) (line 170)
        solve_call_result_236374 = invoke(stypy.reporting.localization.Localization(__file__, 170, 27), solve_236371, *[trust_radius_236372], **kwargs_236373)
        
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___236375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), solve_call_result_236374, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_236376 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), getitem___236375, int_236369)
        
        # Assigning a type to the variable 'tuple_var_assignment_235671' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_235671', subscript_call_result_236376)
        
        # Assigning a Name to a Name (line 170):
        # Getting the type of 'tuple_var_assignment_235670' (line 170)
        tuple_var_assignment_235670_236377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_235670')
        # Assigning a type to the variable 'p' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'p', tuple_var_assignment_235670_236377)
        
        # Assigning a Name to a Name (line 170):
        # Getting the type of 'tuple_var_assignment_235671' (line 170)
        tuple_var_assignment_235671_236378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_235671')
        # Assigning a type to the variable 'hits_boundary' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'hits_boundary', tuple_var_assignment_235671_236378)
        
        # Assigning a Call to a Tuple (line 171):
        
        # Assigning a Subscript to a Name (line 171):
        
        # Obtaining the type of the subscript
        int_236379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'int')
        
        # Call to readouterr(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_236382 = {}
        # Getting the type of 'capsys' (line 171)
        capsys_236380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 19), 'capsys', False)
        # Obtaining the member 'readouterr' of a type (line 171)
        readouterr_236381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 19), capsys_236380, 'readouterr')
        # Calling readouterr(args, kwargs) (line 171)
        readouterr_call_result_236383 = invoke(stypy.reporting.localization.Localization(__file__, 171, 19), readouterr_236381, *[], **kwargs_236382)
        
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___236384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), readouterr_call_result_236383, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_236385 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), getitem___236384, int_236379)
        
        # Assigning a type to the variable 'tuple_var_assignment_235672' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_235672', subscript_call_result_236385)
        
        # Assigning a Subscript to a Name (line 171):
        
        # Obtaining the type of the subscript
        int_236386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'int')
        
        # Call to readouterr(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_236389 = {}
        # Getting the type of 'capsys' (line 171)
        capsys_236387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 19), 'capsys', False)
        # Obtaining the member 'readouterr' of a type (line 171)
        readouterr_236388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 19), capsys_236387, 'readouterr')
        # Calling readouterr(args, kwargs) (line 171)
        readouterr_call_result_236390 = invoke(stypy.reporting.localization.Localization(__file__, 171, 19), readouterr_236388, *[], **kwargs_236389)
        
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___236391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), readouterr_call_result_236390, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_236392 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), getitem___236391, int_236386)
        
        # Assigning a type to the variable 'tuple_var_assignment_235673' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_235673', subscript_call_result_236392)
        
        # Assigning a Name to a Name (line 171):
        # Getting the type of 'tuple_var_assignment_235672' (line 171)
        tuple_var_assignment_235672_236393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_235672')
        # Assigning a type to the variable 'out' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'out', tuple_var_assignment_235672_236393)
        
        # Assigning a Name to a Name (line 171):
        # Getting the type of 'tuple_var_assignment_235673' (line 171)
        tuple_var_assignment_235673_236394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_235673')
        # Assigning a type to the variable 'err' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 13), 'err', tuple_var_assignment_235673_236394)
        
        # Call to assert_(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Call to startswith(...): (line 172)
        # Processing the call arguments (line 172)
        str_236398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 31), 'str', ' TR Solving trust region problem')
        # Processing the call keyword arguments (line 172)
        kwargs_236399 = {}
        # Getting the type of 'out' (line 172)
        out_236396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'out', False)
        # Obtaining the member 'startswith' of a type (line 172)
        startswith_236397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), out_236396, 'startswith')
        # Calling startswith(args, kwargs) (line 172)
        startswith_call_result_236400 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), startswith_236397, *[str_236398], **kwargs_236399)
        
        
        # Call to repr(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'out' (line 172)
        out_236402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 73), 'out', False)
        # Processing the call keyword arguments (line 172)
        kwargs_236403 = {}
        # Getting the type of 'repr' (line 172)
        repr_236401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 68), 'repr', False)
        # Calling repr(args, kwargs) (line 172)
        repr_call_result_236404 = invoke(stypy.reporting.localization.Localization(__file__, 172, 68), repr_236401, *[out_236402], **kwargs_236403)
        
        # Processing the call keyword arguments (line 172)
        kwargs_236405 = {}
        # Getting the type of 'assert_' (line 172)
        assert__236395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 172)
        assert__call_result_236406 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), assert__236395, *[startswith_call_result_236400, repr_call_result_236404], **kwargs_236405)
        
        
        # ################# End of 'test_disp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_disp' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_236407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_disp'
        return stypy_return_type_236407


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKrylovQuadraticSubproblem.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestKrylovQuadraticSubproblem' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'TestKrylovQuadraticSubproblem', TestKrylovQuadraticSubproblem)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
