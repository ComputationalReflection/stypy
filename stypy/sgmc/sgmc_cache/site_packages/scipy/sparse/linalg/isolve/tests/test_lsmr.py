
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Copyright (C) 2010 David Fong and Michael Saunders
3: Distributed under the same license as Scipy
4: 
5: Testing Code for LSMR.
6: 
7: 03 Jun 2010: First version release with lsmr.py
8: 
9: David Chin-lung Fong            clfong@stanford.edu
10: Institute for Computational and Mathematical Engineering
11: Stanford University
12: 
13: Michael Saunders                saunders@stanford.edu
14: Systems Optimization Laboratory
15: Dept of MS&E, Stanford University.
16: 
17: '''
18: 
19: from __future__ import division, print_function, absolute_import
20: 
21: from numpy import array, arange, eye, zeros, ones, sqrt, transpose, hstack
22: from numpy.linalg import norm
23: from numpy.testing import (assert_almost_equal,
24:                            assert_array_almost_equal)
25: 
26: from scipy.sparse import coo_matrix
27: from scipy.sparse.linalg.interface import aslinearoperator
28: from scipy.sparse.linalg import lsmr
29: from .test_lsqr import G, b
30: 
31: 
32: class TestLSMR:
33:     def setup_method(self):
34:         self.n = 10
35:         self.m = 10
36: 
37:     def assertCompatibleSystem(self, A, xtrue):
38:         Afun = aslinearoperator(A)
39:         b = Afun.matvec(xtrue)
40:         x = lsmr(A, b)[0]
41:         assert_almost_equal(norm(x - xtrue), 0, decimal=5)
42: 
43:     def testIdentityACase1(self):
44:         A = eye(self.n)
45:         xtrue = zeros((self.n, 1))
46:         self.assertCompatibleSystem(A, xtrue)
47: 
48:     def testIdentityACase2(self):
49:         A = eye(self.n)
50:         xtrue = ones((self.n,1))
51:         self.assertCompatibleSystem(A, xtrue)
52: 
53:     def testIdentityACase3(self):
54:         A = eye(self.n)
55:         xtrue = transpose(arange(self.n,0,-1))
56:         self.assertCompatibleSystem(A, xtrue)
57: 
58:     def testBidiagonalA(self):
59:         A = lowerBidiagonalMatrix(20,self.n)
60:         xtrue = transpose(arange(self.n,0,-1))
61:         self.assertCompatibleSystem(A,xtrue)
62: 
63:     def testScalarB(self):
64:         A = array([[1.0, 2.0]])
65:         b = 3.0
66:         x = lsmr(A, b)[0]
67:         assert_almost_equal(norm(A.dot(x) - b), 0)
68: 
69:     def testColumnB(self):
70:         A = eye(self.n)
71:         b = ones((self.n, 1))
72:         x = lsmr(A, b)[0]
73:         assert_almost_equal(norm(A.dot(x) - b.ravel()), 0)
74: 
75:     def testInitialization(self):
76:         # Test that the default setting is not modified
77:         x_ref = lsmr(G, b)[0]
78:         x0 = zeros(b.shape)
79:         x = lsmr(G, b, x0=x0)[0]
80:         assert_array_almost_equal(x_ref, x)
81: 
82:         # Test warm-start with single iteration
83:         x0 = lsmr(G, b, maxiter=1)[0]
84:         x = lsmr(G, b, x0=x0)[0]
85:         assert_array_almost_equal(x_ref, x)
86: 
87: class TestLSMRReturns:
88:     def setup_method(self):
89:         self.n = 10
90:         self.A = lowerBidiagonalMatrix(20,self.n)
91:         self.xtrue = transpose(arange(self.n,0,-1))
92:         self.Afun = aslinearoperator(self.A)
93:         self.b = self.Afun.matvec(self.xtrue)
94:         self.returnValues = lsmr(self.A,self.b)
95: 
96:     def testNormr(self):
97:         x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
98:         assert_almost_equal(normr, norm(self.b - self.Afun.matvec(x)))
99: 
100:     def testNormar(self):
101:         x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
102:         assert_almost_equal(normar,
103:                 norm(self.Afun.rmatvec(self.b - self.Afun.matvec(x))))
104: 
105:     def testNormx(self):
106:         x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
107:         assert_almost_equal(normx, norm(x))
108: 
109: 
110: def lowerBidiagonalMatrix(m, n):
111:     # This is a simple example for testing LSMR.
112:     # It uses the leading m*n submatrix from
113:     # A = [ 1
114:     #       1 2
115:     #         2 3
116:     #           3 4
117:     #             ...
118:     #               n ]
119:     # suitably padded by zeros.
120:     #
121:     # 04 Jun 2010: First version for distribution with lsmr.py
122:     if m <= n:
123:         row = hstack((arange(m, dtype=int),
124:                       arange(1, m, dtype=int)))
125:         col = hstack((arange(m, dtype=int),
126:                       arange(m-1, dtype=int)))
127:         data = hstack((arange(1, m+1, dtype=float),
128:                        arange(1,m, dtype=float)))
129:         return coo_matrix((data, (row, col)), shape=(m,n))
130:     else:
131:         row = hstack((arange(n, dtype=int),
132:                       arange(1, n+1, dtype=int)))
133:         col = hstack((arange(n, dtype=int),
134:                       arange(n, dtype=int)))
135:         data = hstack((arange(1, n+1, dtype=float),
136:                        arange(1,n+1, dtype=float)))
137:         return coo_matrix((data,(row, col)), shape=(m,n))
138: 
139: 
140: def lsmrtest(m, n, damp):
141:     '''Verbose testing of lsmr'''
142: 
143:     A = lowerBidiagonalMatrix(m,n)
144:     xtrue = arange(n,0,-1, dtype=float)
145:     Afun = aslinearoperator(A)
146: 
147:     b = Afun.matvec(xtrue)
148: 
149:     atol = 1.0e-7
150:     btol = 1.0e-7
151:     conlim = 1.0e+10
152:     itnlim = 10*n
153:     show = 1
154: 
155:     x, istop, itn, normr, normar, norma, conda, normx \
156:       = lsmr(A, b, damp, atol, btol, conlim, itnlim, show)
157: 
158:     j1 = min(n,5)
159:     j2 = max(n-4,1)
160:     print(' ')
161:     print('First elements of x:')
162:     str = ['%10.4f' % (xi) for xi in x[0:j1]]
163:     print(''.join(str))
164:     print(' ')
165:     print('Last  elements of x:')
166:     str = ['%10.4f' % (xi) for xi in x[j2-1:]]
167:     print(''.join(str))
168: 
169:     r = b - Afun.matvec(x)
170:     r2 = sqrt(norm(r)**2 + (damp*norm(x))**2)
171:     print(' ')
172:     str = 'normr (est.)  %17.10e' % (normr)
173:     str2 = 'normr (true)  %17.10e' % (r2)
174:     print(str)
175:     print(str2)
176:     print(' ')
177: 
178: if __name__ == "__main__":
179:     lsmrtest(20,10,0)
180: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_420830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nCopyright (C) 2010 David Fong and Michael Saunders\nDistributed under the same license as Scipy\n\nTesting Code for LSMR.\n\n03 Jun 2010: First version release with lsmr.py\n\nDavid Chin-lung Fong            clfong@stanford.edu\nInstitute for Computational and Mathematical Engineering\nStanford University\n\nMichael Saunders                saunders@stanford.edu\nSystems Optimization Laboratory\nDept of MS&E, Stanford University.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy import array, arange, eye, zeros, ones, sqrt, transpose, hstack' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_420831 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy')

if (type(import_420831) is not StypyTypeError):

    if (import_420831 != 'pyd_module'):
        __import__(import_420831)
        sys_modules_420832 = sys.modules[import_420831]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', sys_modules_420832.module_type_store, module_type_store, ['array', 'arange', 'eye', 'zeros', 'ones', 'sqrt', 'transpose', 'hstack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_420832, sys_modules_420832.module_type_store, module_type_store)
    else:
        from numpy import array, arange, eye, zeros, ones, sqrt, transpose, hstack

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', None, module_type_store, ['array', 'arange', 'eye', 'zeros', 'ones', 'sqrt', 'transpose', 'hstack'], [array, arange, eye, zeros, ones, sqrt, transpose, hstack])

else:
    # Assigning a type to the variable 'numpy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', import_420831)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.linalg import norm' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_420833 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.linalg')

if (type(import_420833) is not StypyTypeError):

    if (import_420833 != 'pyd_module'):
        __import__(import_420833)
        sys_modules_420834 = sys.modules[import_420833]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.linalg', sys_modules_420834.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_420834, sys_modules_420834.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.linalg', import_420833)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.testing import assert_almost_equal, assert_array_almost_equal' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_420835 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.testing')

if (type(import_420835) is not StypyTypeError):

    if (import_420835 != 'pyd_module'):
        __import__(import_420835)
        sys_modules_420836 = sys.modules[import_420835]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.testing', sys_modules_420836.module_type_store, module_type_store, ['assert_almost_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_420836, sys_modules_420836.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_array_almost_equal'], [assert_almost_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.testing', import_420835)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy.sparse import coo_matrix' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_420837 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse')

if (type(import_420837) is not StypyTypeError):

    if (import_420837 != 'pyd_module'):
        __import__(import_420837)
        sys_modules_420838 = sys.modules[import_420837]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', sys_modules_420838.module_type_store, module_type_store, ['coo_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_420838, sys_modules_420838.module_type_store, module_type_store)
    else:
        from scipy.sparse import coo_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', None, module_type_store, ['coo_matrix'], [coo_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', import_420837)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from scipy.sparse.linalg.interface import aslinearoperator' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_420839 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg.interface')

if (type(import_420839) is not StypyTypeError):

    if (import_420839 != 'pyd_module'):
        __import__(import_420839)
        sys_modules_420840 = sys.modules[import_420839]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg.interface', sys_modules_420840.module_type_store, module_type_store, ['aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_420840, sys_modules_420840.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg.interface', import_420839)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from scipy.sparse.linalg import lsmr' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_420841 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg')

if (type(import_420841) is not StypyTypeError):

    if (import_420841 != 'pyd_module'):
        __import__(import_420841)
        sys_modules_420842 = sys.modules[import_420841]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg', sys_modules_420842.module_type_store, module_type_store, ['lsmr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_420842, sys_modules_420842.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import lsmr

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg', None, module_type_store, ['lsmr'], [lsmr])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg', import_420841)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from scipy.sparse.linalg.isolve.tests.test_lsqr import G, b' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_420843 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.sparse.linalg.isolve.tests.test_lsqr')

if (type(import_420843) is not StypyTypeError):

    if (import_420843 != 'pyd_module'):
        __import__(import_420843)
        sys_modules_420844 = sys.modules[import_420843]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.sparse.linalg.isolve.tests.test_lsqr', sys_modules_420844.module_type_store, module_type_store, ['G', 'b'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_420844, sys_modules_420844.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.tests.test_lsqr import G, b

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.sparse.linalg.isolve.tests.test_lsqr', None, module_type_store, ['G', 'b'], [G, b])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.tests.test_lsqr' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.sparse.linalg.isolve.tests.test_lsqr', import_420843)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

# Declaration of the 'TestLSMR' class

class TestLSMR:

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.setup_method.__dict__.__setitem__('stypy_function_name', 'TestLSMR.setup_method')
        TestLSMR.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMR.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 34):
        
        # Assigning a Num to a Attribute (line 34):
        int_420845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'int')
        # Getting the type of 'self' (line 34)
        self_420846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'n' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_420846, 'n', int_420845)
        
        # Assigning a Num to a Attribute (line 35):
        
        # Assigning a Num to a Attribute (line 35):
        int_420847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'int')
        # Getting the type of 'self' (line 35)
        self_420848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'm' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_420848, 'm', int_420847)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_420849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_420849


    @norecursion
    def assertCompatibleSystem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'assertCompatibleSystem'
        module_type_store = module_type_store.open_function_context('assertCompatibleSystem', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_function_name', 'TestLSMR.assertCompatibleSystem')
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_param_names_list', ['A', 'xtrue'])
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.assertCompatibleSystem.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.assertCompatibleSystem', ['A', 'xtrue'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertCompatibleSystem', localization, ['A', 'xtrue'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertCompatibleSystem(...)' code ##################

        
        # Assigning a Call to a Name (line 38):
        
        # Assigning a Call to a Name (line 38):
        
        # Call to aslinearoperator(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'A' (line 38)
        A_420851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'A', False)
        # Processing the call keyword arguments (line 38)
        kwargs_420852 = {}
        # Getting the type of 'aslinearoperator' (line 38)
        aslinearoperator_420850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'aslinearoperator', False)
        # Calling aslinearoperator(args, kwargs) (line 38)
        aslinearoperator_call_result_420853 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), aslinearoperator_420850, *[A_420851], **kwargs_420852)
        
        # Assigning a type to the variable 'Afun' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'Afun', aslinearoperator_call_result_420853)
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to matvec(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'xtrue' (line 39)
        xtrue_420856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'xtrue', False)
        # Processing the call keyword arguments (line 39)
        kwargs_420857 = {}
        # Getting the type of 'Afun' (line 39)
        Afun_420854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'Afun', False)
        # Obtaining the member 'matvec' of a type (line 39)
        matvec_420855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), Afun_420854, 'matvec')
        # Calling matvec(args, kwargs) (line 39)
        matvec_call_result_420858 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), matvec_420855, *[xtrue_420856], **kwargs_420857)
        
        # Assigning a type to the variable 'b' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'b', matvec_call_result_420858)
        
        # Assigning a Subscript to a Name (line 40):
        
        # Assigning a Subscript to a Name (line 40):
        
        # Obtaining the type of the subscript
        int_420859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'int')
        
        # Call to lsmr(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'A' (line 40)
        A_420861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'A', False)
        # Getting the type of 'b' (line 40)
        b_420862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'b', False)
        # Processing the call keyword arguments (line 40)
        kwargs_420863 = {}
        # Getting the type of 'lsmr' (line 40)
        lsmr_420860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'lsmr', False)
        # Calling lsmr(args, kwargs) (line 40)
        lsmr_call_result_420864 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), lsmr_420860, *[A_420861, b_420862], **kwargs_420863)
        
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___420865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), lsmr_call_result_420864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_420866 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), getitem___420865, int_420859)
        
        # Assigning a type to the variable 'x' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'x', subscript_call_result_420866)
        
        # Call to assert_almost_equal(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to norm(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'x' (line 41)
        x_420869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 33), 'x', False)
        # Getting the type of 'xtrue' (line 41)
        xtrue_420870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'xtrue', False)
        # Applying the binary operator '-' (line 41)
        result_sub_420871 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 33), '-', x_420869, xtrue_420870)
        
        # Processing the call keyword arguments (line 41)
        kwargs_420872 = {}
        # Getting the type of 'norm' (line 41)
        norm_420868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'norm', False)
        # Calling norm(args, kwargs) (line 41)
        norm_call_result_420873 = invoke(stypy.reporting.localization.Localization(__file__, 41, 28), norm_420868, *[result_sub_420871], **kwargs_420872)
        
        int_420874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 45), 'int')
        # Processing the call keyword arguments (line 41)
        int_420875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 56), 'int')
        keyword_420876 = int_420875
        kwargs_420877 = {'decimal': keyword_420876}
        # Getting the type of 'assert_almost_equal' (line 41)
        assert_almost_equal_420867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 41)
        assert_almost_equal_call_result_420878 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assert_almost_equal_420867, *[norm_call_result_420873, int_420874], **kwargs_420877)
        
        
        # ################# End of 'assertCompatibleSystem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertCompatibleSystem' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_420879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420879)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertCompatibleSystem'
        return stypy_return_type_420879


    @norecursion
    def testIdentityACase1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testIdentityACase1'
        module_type_store = module_type_store.open_function_context('testIdentityACase1', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_function_name', 'TestLSMR.testIdentityACase1')
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.testIdentityACase1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.testIdentityACase1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testIdentityACase1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testIdentityACase1(...)' code ##################

        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to eye(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'self' (line 44)
        self_420881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'self', False)
        # Obtaining the member 'n' of a type (line 44)
        n_420882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), self_420881, 'n')
        # Processing the call keyword arguments (line 44)
        kwargs_420883 = {}
        # Getting the type of 'eye' (line 44)
        eye_420880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 44)
        eye_call_result_420884 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), eye_420880, *[n_420882], **kwargs_420883)
        
        # Assigning a type to the variable 'A' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'A', eye_call_result_420884)
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to zeros(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Obtaining an instance of the builtin type 'tuple' (line 45)
        tuple_420886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 45)
        # Adding element type (line 45)
        # Getting the type of 'self' (line 45)
        self_420887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'self', False)
        # Obtaining the member 'n' of a type (line 45)
        n_420888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 23), self_420887, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 23), tuple_420886, n_420888)
        # Adding element type (line 45)
        int_420889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 23), tuple_420886, int_420889)
        
        # Processing the call keyword arguments (line 45)
        kwargs_420890 = {}
        # Getting the type of 'zeros' (line 45)
        zeros_420885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 45)
        zeros_call_result_420891 = invoke(stypy.reporting.localization.Localization(__file__, 45, 16), zeros_420885, *[tuple_420886], **kwargs_420890)
        
        # Assigning a type to the variable 'xtrue' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'xtrue', zeros_call_result_420891)
        
        # Call to assertCompatibleSystem(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'A' (line 46)
        A_420894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'A', False)
        # Getting the type of 'xtrue' (line 46)
        xtrue_420895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), 'xtrue', False)
        # Processing the call keyword arguments (line 46)
        kwargs_420896 = {}
        # Getting the type of 'self' (line 46)
        self_420892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member 'assertCompatibleSystem' of a type (line 46)
        assertCompatibleSystem_420893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_420892, 'assertCompatibleSystem')
        # Calling assertCompatibleSystem(args, kwargs) (line 46)
        assertCompatibleSystem_call_result_420897 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assertCompatibleSystem_420893, *[A_420894, xtrue_420895], **kwargs_420896)
        
        
        # ################# End of 'testIdentityACase1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testIdentityACase1' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_420898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testIdentityACase1'
        return stypy_return_type_420898


    @norecursion
    def testIdentityACase2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testIdentityACase2'
        module_type_store = module_type_store.open_function_context('testIdentityACase2', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_function_name', 'TestLSMR.testIdentityACase2')
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.testIdentityACase2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.testIdentityACase2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testIdentityACase2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testIdentityACase2(...)' code ##################

        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to eye(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'self' (line 49)
        self_420900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'self', False)
        # Obtaining the member 'n' of a type (line 49)
        n_420901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), self_420900, 'n')
        # Processing the call keyword arguments (line 49)
        kwargs_420902 = {}
        # Getting the type of 'eye' (line 49)
        eye_420899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 49)
        eye_call_result_420903 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), eye_420899, *[n_420901], **kwargs_420902)
        
        # Assigning a type to the variable 'A' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'A', eye_call_result_420903)
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to ones(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_420905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_420906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'self', False)
        # Obtaining the member 'n' of a type (line 50)
        n_420907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 22), self_420906, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 22), tuple_420905, n_420907)
        # Adding element type (line 50)
        int_420908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 22), tuple_420905, int_420908)
        
        # Processing the call keyword arguments (line 50)
        kwargs_420909 = {}
        # Getting the type of 'ones' (line 50)
        ones_420904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'ones', False)
        # Calling ones(args, kwargs) (line 50)
        ones_call_result_420910 = invoke(stypy.reporting.localization.Localization(__file__, 50, 16), ones_420904, *[tuple_420905], **kwargs_420909)
        
        # Assigning a type to the variable 'xtrue' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'xtrue', ones_call_result_420910)
        
        # Call to assertCompatibleSystem(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'A' (line 51)
        A_420913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 36), 'A', False)
        # Getting the type of 'xtrue' (line 51)
        xtrue_420914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 39), 'xtrue', False)
        # Processing the call keyword arguments (line 51)
        kwargs_420915 = {}
        # Getting the type of 'self' (line 51)
        self_420911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self', False)
        # Obtaining the member 'assertCompatibleSystem' of a type (line 51)
        assertCompatibleSystem_420912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_420911, 'assertCompatibleSystem')
        # Calling assertCompatibleSystem(args, kwargs) (line 51)
        assertCompatibleSystem_call_result_420916 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assertCompatibleSystem_420912, *[A_420913, xtrue_420914], **kwargs_420915)
        
        
        # ################# End of 'testIdentityACase2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testIdentityACase2' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_420917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420917)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testIdentityACase2'
        return stypy_return_type_420917


    @norecursion
    def testIdentityACase3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testIdentityACase3'
        module_type_store = module_type_store.open_function_context('testIdentityACase3', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_function_name', 'TestLSMR.testIdentityACase3')
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.testIdentityACase3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.testIdentityACase3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testIdentityACase3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testIdentityACase3(...)' code ##################

        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to eye(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_420919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'self', False)
        # Obtaining the member 'n' of a type (line 54)
        n_420920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), self_420919, 'n')
        # Processing the call keyword arguments (line 54)
        kwargs_420921 = {}
        # Getting the type of 'eye' (line 54)
        eye_420918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 54)
        eye_call_result_420922 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), eye_420918, *[n_420920], **kwargs_420921)
        
        # Assigning a type to the variable 'A' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'A', eye_call_result_420922)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to transpose(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Call to arange(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_420925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'self', False)
        # Obtaining the member 'n' of a type (line 55)
        n_420926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 33), self_420925, 'n')
        int_420927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'int')
        int_420928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'int')
        # Processing the call keyword arguments (line 55)
        kwargs_420929 = {}
        # Getting the type of 'arange' (line 55)
        arange_420924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 26), 'arange', False)
        # Calling arange(args, kwargs) (line 55)
        arange_call_result_420930 = invoke(stypy.reporting.localization.Localization(__file__, 55, 26), arange_420924, *[n_420926, int_420927, int_420928], **kwargs_420929)
        
        # Processing the call keyword arguments (line 55)
        kwargs_420931 = {}
        # Getting the type of 'transpose' (line 55)
        transpose_420923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'transpose', False)
        # Calling transpose(args, kwargs) (line 55)
        transpose_call_result_420932 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), transpose_420923, *[arange_call_result_420930], **kwargs_420931)
        
        # Assigning a type to the variable 'xtrue' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'xtrue', transpose_call_result_420932)
        
        # Call to assertCompatibleSystem(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'A' (line 56)
        A_420935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'A', False)
        # Getting the type of 'xtrue' (line 56)
        xtrue_420936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'xtrue', False)
        # Processing the call keyword arguments (line 56)
        kwargs_420937 = {}
        # Getting the type of 'self' (line 56)
        self_420933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self', False)
        # Obtaining the member 'assertCompatibleSystem' of a type (line 56)
        assertCompatibleSystem_420934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_420933, 'assertCompatibleSystem')
        # Calling assertCompatibleSystem(args, kwargs) (line 56)
        assertCompatibleSystem_call_result_420938 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assertCompatibleSystem_420934, *[A_420935, xtrue_420936], **kwargs_420937)
        
        
        # ################# End of 'testIdentityACase3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testIdentityACase3' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_420939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420939)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testIdentityACase3'
        return stypy_return_type_420939


    @norecursion
    def testBidiagonalA(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBidiagonalA'
        module_type_store = module_type_store.open_function_context('testBidiagonalA', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_function_name', 'TestLSMR.testBidiagonalA')
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.testBidiagonalA.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.testBidiagonalA', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBidiagonalA', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBidiagonalA(...)' code ##################

        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to lowerBidiagonalMatrix(...): (line 59)
        # Processing the call arguments (line 59)
        int_420941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 34), 'int')
        # Getting the type of 'self' (line 59)
        self_420942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'self', False)
        # Obtaining the member 'n' of a type (line 59)
        n_420943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 37), self_420942, 'n')
        # Processing the call keyword arguments (line 59)
        kwargs_420944 = {}
        # Getting the type of 'lowerBidiagonalMatrix' (line 59)
        lowerBidiagonalMatrix_420940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'lowerBidiagonalMatrix', False)
        # Calling lowerBidiagonalMatrix(args, kwargs) (line 59)
        lowerBidiagonalMatrix_call_result_420945 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), lowerBidiagonalMatrix_420940, *[int_420941, n_420943], **kwargs_420944)
        
        # Assigning a type to the variable 'A' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'A', lowerBidiagonalMatrix_call_result_420945)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to transpose(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to arange(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_420948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 33), 'self', False)
        # Obtaining the member 'n' of a type (line 60)
        n_420949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 33), self_420948, 'n')
        int_420950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 40), 'int')
        int_420951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 42), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_420952 = {}
        # Getting the type of 'arange' (line 60)
        arange_420947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'arange', False)
        # Calling arange(args, kwargs) (line 60)
        arange_call_result_420953 = invoke(stypy.reporting.localization.Localization(__file__, 60, 26), arange_420947, *[n_420949, int_420950, int_420951], **kwargs_420952)
        
        # Processing the call keyword arguments (line 60)
        kwargs_420954 = {}
        # Getting the type of 'transpose' (line 60)
        transpose_420946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'transpose', False)
        # Calling transpose(args, kwargs) (line 60)
        transpose_call_result_420955 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), transpose_420946, *[arange_call_result_420953], **kwargs_420954)
        
        # Assigning a type to the variable 'xtrue' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'xtrue', transpose_call_result_420955)
        
        # Call to assertCompatibleSystem(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'A' (line 61)
        A_420958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 36), 'A', False)
        # Getting the type of 'xtrue' (line 61)
        xtrue_420959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'xtrue', False)
        # Processing the call keyword arguments (line 61)
        kwargs_420960 = {}
        # Getting the type of 'self' (line 61)
        self_420956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'assertCompatibleSystem' of a type (line 61)
        assertCompatibleSystem_420957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_420956, 'assertCompatibleSystem')
        # Calling assertCompatibleSystem(args, kwargs) (line 61)
        assertCompatibleSystem_call_result_420961 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assertCompatibleSystem_420957, *[A_420958, xtrue_420959], **kwargs_420960)
        
        
        # ################# End of 'testBidiagonalA(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBidiagonalA' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_420962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420962)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBidiagonalA'
        return stypy_return_type_420962


    @norecursion
    def testScalarB(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testScalarB'
        module_type_store = module_type_store.open_function_context('testScalarB', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_function_name', 'TestLSMR.testScalarB')
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.testScalarB.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.testScalarB', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testScalarB', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testScalarB(...)' code ##################

        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to array(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_420964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_420965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        float_420966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_420965, float_420966)
        # Adding element type (line 64)
        float_420967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_420965, float_420967)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 18), list_420964, list_420965)
        
        # Processing the call keyword arguments (line 64)
        kwargs_420968 = {}
        # Getting the type of 'array' (line 64)
        array_420963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'array', False)
        # Calling array(args, kwargs) (line 64)
        array_call_result_420969 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), array_420963, *[list_420964], **kwargs_420968)
        
        # Assigning a type to the variable 'A' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'A', array_call_result_420969)
        
        # Assigning a Num to a Name (line 65):
        
        # Assigning a Num to a Name (line 65):
        float_420970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'float')
        # Assigning a type to the variable 'b' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'b', float_420970)
        
        # Assigning a Subscript to a Name (line 66):
        
        # Assigning a Subscript to a Name (line 66):
        
        # Obtaining the type of the subscript
        int_420971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'int')
        
        # Call to lsmr(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'A' (line 66)
        A_420973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'A', False)
        # Getting the type of 'b' (line 66)
        b_420974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'b', False)
        # Processing the call keyword arguments (line 66)
        kwargs_420975 = {}
        # Getting the type of 'lsmr' (line 66)
        lsmr_420972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'lsmr', False)
        # Calling lsmr(args, kwargs) (line 66)
        lsmr_call_result_420976 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), lsmr_420972, *[A_420973, b_420974], **kwargs_420975)
        
        # Obtaining the member '__getitem__' of a type (line 66)
        getitem___420977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), lsmr_call_result_420976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 66)
        subscript_call_result_420978 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), getitem___420977, int_420971)
        
        # Assigning a type to the variable 'x' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'x', subscript_call_result_420978)
        
        # Call to assert_almost_equal(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to norm(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to dot(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'x' (line 67)
        x_420983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'x', False)
        # Processing the call keyword arguments (line 67)
        kwargs_420984 = {}
        # Getting the type of 'A' (line 67)
        A_420981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'A', False)
        # Obtaining the member 'dot' of a type (line 67)
        dot_420982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 33), A_420981, 'dot')
        # Calling dot(args, kwargs) (line 67)
        dot_call_result_420985 = invoke(stypy.reporting.localization.Localization(__file__, 67, 33), dot_420982, *[x_420983], **kwargs_420984)
        
        # Getting the type of 'b' (line 67)
        b_420986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 44), 'b', False)
        # Applying the binary operator '-' (line 67)
        result_sub_420987 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 33), '-', dot_call_result_420985, b_420986)
        
        # Processing the call keyword arguments (line 67)
        kwargs_420988 = {}
        # Getting the type of 'norm' (line 67)
        norm_420980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'norm', False)
        # Calling norm(args, kwargs) (line 67)
        norm_call_result_420989 = invoke(stypy.reporting.localization.Localization(__file__, 67, 28), norm_420980, *[result_sub_420987], **kwargs_420988)
        
        int_420990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'int')
        # Processing the call keyword arguments (line 67)
        kwargs_420991 = {}
        # Getting the type of 'assert_almost_equal' (line 67)
        assert_almost_equal_420979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 67)
        assert_almost_equal_call_result_420992 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assert_almost_equal_420979, *[norm_call_result_420989, int_420990], **kwargs_420991)
        
        
        # ################# End of 'testScalarB(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testScalarB' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_420993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420993)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testScalarB'
        return stypy_return_type_420993


    @norecursion
    def testColumnB(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testColumnB'
        module_type_store = module_type_store.open_function_context('testColumnB', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_function_name', 'TestLSMR.testColumnB')
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.testColumnB.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.testColumnB', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testColumnB', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testColumnB(...)' code ##################

        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to eye(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_420995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'self', False)
        # Obtaining the member 'n' of a type (line 70)
        n_420996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), self_420995, 'n')
        # Processing the call keyword arguments (line 70)
        kwargs_420997 = {}
        # Getting the type of 'eye' (line 70)
        eye_420994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 70)
        eye_call_result_420998 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), eye_420994, *[n_420996], **kwargs_420997)
        
        # Assigning a type to the variable 'A' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'A', eye_call_result_420998)
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to ones(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_421000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        # Getting the type of 'self' (line 71)
        self_421001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'self', False)
        # Obtaining the member 'n' of a type (line 71)
        n_421002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 18), self_421001, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), tuple_421000, n_421002)
        # Adding element type (line 71)
        int_421003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), tuple_421000, int_421003)
        
        # Processing the call keyword arguments (line 71)
        kwargs_421004 = {}
        # Getting the type of 'ones' (line 71)
        ones_420999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 71)
        ones_call_result_421005 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), ones_420999, *[tuple_421000], **kwargs_421004)
        
        # Assigning a type to the variable 'b' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'b', ones_call_result_421005)
        
        # Assigning a Subscript to a Name (line 72):
        
        # Assigning a Subscript to a Name (line 72):
        
        # Obtaining the type of the subscript
        int_421006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 23), 'int')
        
        # Call to lsmr(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'A' (line 72)
        A_421008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'A', False)
        # Getting the type of 'b' (line 72)
        b_421009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'b', False)
        # Processing the call keyword arguments (line 72)
        kwargs_421010 = {}
        # Getting the type of 'lsmr' (line 72)
        lsmr_421007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'lsmr', False)
        # Calling lsmr(args, kwargs) (line 72)
        lsmr_call_result_421011 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), lsmr_421007, *[A_421008, b_421009], **kwargs_421010)
        
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___421012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), lsmr_call_result_421011, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_421013 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), getitem___421012, int_421006)
        
        # Assigning a type to the variable 'x' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'x', subscript_call_result_421013)
        
        # Call to assert_almost_equal(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to norm(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to dot(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'x' (line 73)
        x_421018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 39), 'x', False)
        # Processing the call keyword arguments (line 73)
        kwargs_421019 = {}
        # Getting the type of 'A' (line 73)
        A_421016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 33), 'A', False)
        # Obtaining the member 'dot' of a type (line 73)
        dot_421017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 33), A_421016, 'dot')
        # Calling dot(args, kwargs) (line 73)
        dot_call_result_421020 = invoke(stypy.reporting.localization.Localization(__file__, 73, 33), dot_421017, *[x_421018], **kwargs_421019)
        
        
        # Call to ravel(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_421023 = {}
        # Getting the type of 'b' (line 73)
        b_421021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'b', False)
        # Obtaining the member 'ravel' of a type (line 73)
        ravel_421022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 44), b_421021, 'ravel')
        # Calling ravel(args, kwargs) (line 73)
        ravel_call_result_421024 = invoke(stypy.reporting.localization.Localization(__file__, 73, 44), ravel_421022, *[], **kwargs_421023)
        
        # Applying the binary operator '-' (line 73)
        result_sub_421025 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 33), '-', dot_call_result_421020, ravel_call_result_421024)
        
        # Processing the call keyword arguments (line 73)
        kwargs_421026 = {}
        # Getting the type of 'norm' (line 73)
        norm_421015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'norm', False)
        # Calling norm(args, kwargs) (line 73)
        norm_call_result_421027 = invoke(stypy.reporting.localization.Localization(__file__, 73, 28), norm_421015, *[result_sub_421025], **kwargs_421026)
        
        int_421028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 56), 'int')
        # Processing the call keyword arguments (line 73)
        kwargs_421029 = {}
        # Getting the type of 'assert_almost_equal' (line 73)
        assert_almost_equal_421014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 73)
        assert_almost_equal_call_result_421030 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assert_almost_equal_421014, *[norm_call_result_421027, int_421028], **kwargs_421029)
        
        
        # ################# End of 'testColumnB(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testColumnB' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_421031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_421031)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testColumnB'
        return stypy_return_type_421031


    @norecursion
    def testInitialization(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testInitialization'
        module_type_store = module_type_store.open_function_context('testInitialization', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_localization', localization)
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_function_name', 'TestLSMR.testInitialization')
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMR.testInitialization.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.testInitialization', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testInitialization', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testInitialization(...)' code ##################

        
        # Assigning a Subscript to a Name (line 77):
        
        # Assigning a Subscript to a Name (line 77):
        
        # Obtaining the type of the subscript
        int_421032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
        
        # Call to lsmr(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'G' (line 77)
        G_421034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'G', False)
        # Getting the type of 'b' (line 77)
        b_421035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'b', False)
        # Processing the call keyword arguments (line 77)
        kwargs_421036 = {}
        # Getting the type of 'lsmr' (line 77)
        lsmr_421033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'lsmr', False)
        # Calling lsmr(args, kwargs) (line 77)
        lsmr_call_result_421037 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), lsmr_421033, *[G_421034, b_421035], **kwargs_421036)
        
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___421038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), lsmr_call_result_421037, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_421039 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), getitem___421038, int_421032)
        
        # Assigning a type to the variable 'x_ref' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'x_ref', subscript_call_result_421039)
        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to zeros(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'b' (line 78)
        b_421041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'b', False)
        # Obtaining the member 'shape' of a type (line 78)
        shape_421042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), b_421041, 'shape')
        # Processing the call keyword arguments (line 78)
        kwargs_421043 = {}
        # Getting the type of 'zeros' (line 78)
        zeros_421040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'zeros', False)
        # Calling zeros(args, kwargs) (line 78)
        zeros_call_result_421044 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), zeros_421040, *[shape_421042], **kwargs_421043)
        
        # Assigning a type to the variable 'x0' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'x0', zeros_call_result_421044)
        
        # Assigning a Subscript to a Name (line 79):
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_421045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'int')
        
        # Call to lsmr(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'G' (line 79)
        G_421047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'G', False)
        # Getting the type of 'b' (line 79)
        b_421048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'b', False)
        # Processing the call keyword arguments (line 79)
        # Getting the type of 'x0' (line 79)
        x0_421049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'x0', False)
        keyword_421050 = x0_421049
        kwargs_421051 = {'x0': keyword_421050}
        # Getting the type of 'lsmr' (line 79)
        lsmr_421046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'lsmr', False)
        # Calling lsmr(args, kwargs) (line 79)
        lsmr_call_result_421052 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), lsmr_421046, *[G_421047, b_421048], **kwargs_421051)
        
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___421053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), lsmr_call_result_421052, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_421054 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), getitem___421053, int_421045)
        
        # Assigning a type to the variable 'x' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'x', subscript_call_result_421054)
        
        # Call to assert_array_almost_equal(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'x_ref' (line 80)
        x_ref_421056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'x_ref', False)
        # Getting the type of 'x' (line 80)
        x_421057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 41), 'x', False)
        # Processing the call keyword arguments (line 80)
        kwargs_421058 = {}
        # Getting the type of 'assert_array_almost_equal' (line 80)
        assert_array_almost_equal_421055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 80)
        assert_array_almost_equal_call_result_421059 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_array_almost_equal_421055, *[x_ref_421056, x_421057], **kwargs_421058)
        
        
        # Assigning a Subscript to a Name (line 83):
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_421060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 35), 'int')
        
        # Call to lsmr(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'G' (line 83)
        G_421062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'G', False)
        # Getting the type of 'b' (line 83)
        b_421063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 21), 'b', False)
        # Processing the call keyword arguments (line 83)
        int_421064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 32), 'int')
        keyword_421065 = int_421064
        kwargs_421066 = {'maxiter': keyword_421065}
        # Getting the type of 'lsmr' (line 83)
        lsmr_421061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'lsmr', False)
        # Calling lsmr(args, kwargs) (line 83)
        lsmr_call_result_421067 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), lsmr_421061, *[G_421062, b_421063], **kwargs_421066)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___421068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), lsmr_call_result_421067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_421069 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), getitem___421068, int_421060)
        
        # Assigning a type to the variable 'x0' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'x0', subscript_call_result_421069)
        
        # Assigning a Subscript to a Name (line 84):
        
        # Assigning a Subscript to a Name (line 84):
        
        # Obtaining the type of the subscript
        int_421070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'int')
        
        # Call to lsmr(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'G' (line 84)
        G_421072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'G', False)
        # Getting the type of 'b' (line 84)
        b_421073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'b', False)
        # Processing the call keyword arguments (line 84)
        # Getting the type of 'x0' (line 84)
        x0_421074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'x0', False)
        keyword_421075 = x0_421074
        kwargs_421076 = {'x0': keyword_421075}
        # Getting the type of 'lsmr' (line 84)
        lsmr_421071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'lsmr', False)
        # Calling lsmr(args, kwargs) (line 84)
        lsmr_call_result_421077 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), lsmr_421071, *[G_421072, b_421073], **kwargs_421076)
        
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___421078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), lsmr_call_result_421077, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_421079 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), getitem___421078, int_421070)
        
        # Assigning a type to the variable 'x' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'x', subscript_call_result_421079)
        
        # Call to assert_array_almost_equal(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'x_ref' (line 85)
        x_ref_421081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'x_ref', False)
        # Getting the type of 'x' (line 85)
        x_421082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'x', False)
        # Processing the call keyword arguments (line 85)
        kwargs_421083 = {}
        # Getting the type of 'assert_array_almost_equal' (line 85)
        assert_array_almost_equal_421080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 85)
        assert_array_almost_equal_call_result_421084 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assert_array_almost_equal_421080, *[x_ref_421081, x_421082], **kwargs_421083)
        
        
        # ################# End of 'testInitialization(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testInitialization' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_421085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_421085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testInitialization'
        return stypy_return_type_421085


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 32, 0, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMR.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLSMR' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'TestLSMR', TestLSMR)
# Declaration of the 'TestLSMRReturns' class

class TestLSMRReturns:

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_function_name', 'TestLSMRReturns.setup_method')
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMRReturns.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMRReturns.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 89):
        
        # Assigning a Num to a Attribute (line 89):
        int_421086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'int')
        # Getting the type of 'self' (line 89)
        self_421087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member 'n' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_421087, 'n', int_421086)
        
        # Assigning a Call to a Attribute (line 90):
        
        # Assigning a Call to a Attribute (line 90):
        
        # Call to lowerBidiagonalMatrix(...): (line 90)
        # Processing the call arguments (line 90)
        int_421089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 39), 'int')
        # Getting the type of 'self' (line 90)
        self_421090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 42), 'self', False)
        # Obtaining the member 'n' of a type (line 90)
        n_421091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 42), self_421090, 'n')
        # Processing the call keyword arguments (line 90)
        kwargs_421092 = {}
        # Getting the type of 'lowerBidiagonalMatrix' (line 90)
        lowerBidiagonalMatrix_421088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'lowerBidiagonalMatrix', False)
        # Calling lowerBidiagonalMatrix(args, kwargs) (line 90)
        lowerBidiagonalMatrix_call_result_421093 = invoke(stypy.reporting.localization.Localization(__file__, 90, 17), lowerBidiagonalMatrix_421088, *[int_421089, n_421091], **kwargs_421092)
        
        # Getting the type of 'self' (line 90)
        self_421094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'A' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_421094, 'A', lowerBidiagonalMatrix_call_result_421093)
        
        # Assigning a Call to a Attribute (line 91):
        
        # Assigning a Call to a Attribute (line 91):
        
        # Call to transpose(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to arange(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'self' (line 91)
        self_421097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'self', False)
        # Obtaining the member 'n' of a type (line 91)
        n_421098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), self_421097, 'n')
        int_421099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 45), 'int')
        int_421100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 47), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_421101 = {}
        # Getting the type of 'arange' (line 91)
        arange_421096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'arange', False)
        # Calling arange(args, kwargs) (line 91)
        arange_call_result_421102 = invoke(stypy.reporting.localization.Localization(__file__, 91, 31), arange_421096, *[n_421098, int_421099, int_421100], **kwargs_421101)
        
        # Processing the call keyword arguments (line 91)
        kwargs_421103 = {}
        # Getting the type of 'transpose' (line 91)
        transpose_421095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'transpose', False)
        # Calling transpose(args, kwargs) (line 91)
        transpose_call_result_421104 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), transpose_421095, *[arange_call_result_421102], **kwargs_421103)
        
        # Getting the type of 'self' (line 91)
        self_421105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'xtrue' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_421105, 'xtrue', transpose_call_result_421104)
        
        # Assigning a Call to a Attribute (line 92):
        
        # Assigning a Call to a Attribute (line 92):
        
        # Call to aslinearoperator(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'self' (line 92)
        self_421107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'self', False)
        # Obtaining the member 'A' of a type (line 92)
        A_421108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 37), self_421107, 'A')
        # Processing the call keyword arguments (line 92)
        kwargs_421109 = {}
        # Getting the type of 'aslinearoperator' (line 92)
        aslinearoperator_421106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'aslinearoperator', False)
        # Calling aslinearoperator(args, kwargs) (line 92)
        aslinearoperator_call_result_421110 = invoke(stypy.reporting.localization.Localization(__file__, 92, 20), aslinearoperator_421106, *[A_421108], **kwargs_421109)
        
        # Getting the type of 'self' (line 92)
        self_421111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member 'Afun' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_421111, 'Afun', aslinearoperator_call_result_421110)
        
        # Assigning a Call to a Attribute (line 93):
        
        # Assigning a Call to a Attribute (line 93):
        
        # Call to matvec(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'self' (line 93)
        self_421115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 'self', False)
        # Obtaining the member 'xtrue' of a type (line 93)
        xtrue_421116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 34), self_421115, 'xtrue')
        # Processing the call keyword arguments (line 93)
        kwargs_421117 = {}
        # Getting the type of 'self' (line 93)
        self_421112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'self', False)
        # Obtaining the member 'Afun' of a type (line 93)
        Afun_421113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), self_421112, 'Afun')
        # Obtaining the member 'matvec' of a type (line 93)
        matvec_421114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), Afun_421113, 'matvec')
        # Calling matvec(args, kwargs) (line 93)
        matvec_call_result_421118 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), matvec_421114, *[xtrue_421116], **kwargs_421117)
        
        # Getting the type of 'self' (line 93)
        self_421119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member 'b' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_421119, 'b', matvec_call_result_421118)
        
        # Assigning a Call to a Attribute (line 94):
        
        # Assigning a Call to a Attribute (line 94):
        
        # Call to lsmr(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'self' (line 94)
        self_421121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'self', False)
        # Obtaining the member 'A' of a type (line 94)
        A_421122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), self_421121, 'A')
        # Getting the type of 'self' (line 94)
        self_421123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 40), 'self', False)
        # Obtaining the member 'b' of a type (line 94)
        b_421124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 40), self_421123, 'b')
        # Processing the call keyword arguments (line 94)
        kwargs_421125 = {}
        # Getting the type of 'lsmr' (line 94)
        lsmr_421120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'lsmr', False)
        # Calling lsmr(args, kwargs) (line 94)
        lsmr_call_result_421126 = invoke(stypy.reporting.localization.Localization(__file__, 94, 28), lsmr_421120, *[A_421122, b_421124], **kwargs_421125)
        
        # Getting the type of 'self' (line 94)
        self_421127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'returnValues' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_421127, 'returnValues', lsmr_call_result_421126)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_421128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_421128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_421128


    @norecursion
    def testNormr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testNormr'
        module_type_store = module_type_store.open_function_context('testNormr', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_localization', localization)
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_function_name', 'TestLSMRReturns.testNormr')
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMRReturns.testNormr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMRReturns.testNormr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testNormr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testNormr(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 97):
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_421129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
        # Getting the type of 'self' (line 97)
        self_421130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 97)
        returnValues_421131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 60), self_421130, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___421132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), returnValues_421131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_421133 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___421132, int_421129)
        
        # Assigning a type to the variable 'tuple_var_assignment_420798' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420798', subscript_call_result_421133)
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_421134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
        # Getting the type of 'self' (line 97)
        self_421135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 97)
        returnValues_421136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 60), self_421135, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___421137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), returnValues_421136, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_421138 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___421137, int_421134)
        
        # Assigning a type to the variable 'tuple_var_assignment_420799' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420799', subscript_call_result_421138)
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_421139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
        # Getting the type of 'self' (line 97)
        self_421140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 97)
        returnValues_421141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 60), self_421140, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___421142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), returnValues_421141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_421143 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___421142, int_421139)
        
        # Assigning a type to the variable 'tuple_var_assignment_420800' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420800', subscript_call_result_421143)
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_421144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
        # Getting the type of 'self' (line 97)
        self_421145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 97)
        returnValues_421146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 60), self_421145, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___421147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), returnValues_421146, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_421148 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___421147, int_421144)
        
        # Assigning a type to the variable 'tuple_var_assignment_420801' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420801', subscript_call_result_421148)
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_421149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
        # Getting the type of 'self' (line 97)
        self_421150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 97)
        returnValues_421151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 60), self_421150, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___421152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), returnValues_421151, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_421153 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___421152, int_421149)
        
        # Assigning a type to the variable 'tuple_var_assignment_420802' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420802', subscript_call_result_421153)
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_421154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
        # Getting the type of 'self' (line 97)
        self_421155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 97)
        returnValues_421156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 60), self_421155, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___421157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), returnValues_421156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_421158 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___421157, int_421154)
        
        # Assigning a type to the variable 'tuple_var_assignment_420803' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420803', subscript_call_result_421158)
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_421159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
        # Getting the type of 'self' (line 97)
        self_421160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 97)
        returnValues_421161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 60), self_421160, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___421162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), returnValues_421161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_421163 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___421162, int_421159)
        
        # Assigning a type to the variable 'tuple_var_assignment_420804' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420804', subscript_call_result_421163)
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_421164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
        # Getting the type of 'self' (line 97)
        self_421165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 97)
        returnValues_421166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 60), self_421165, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___421167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), returnValues_421166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_421168 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___421167, int_421164)
        
        # Assigning a type to the variable 'tuple_var_assignment_420805' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420805', subscript_call_result_421168)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_420798' (line 97)
        tuple_var_assignment_420798_421169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420798')
        # Assigning a type to the variable 'x' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'x', tuple_var_assignment_420798_421169)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_420799' (line 97)
        tuple_var_assignment_420799_421170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420799')
        # Assigning a type to the variable 'istop' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'istop', tuple_var_assignment_420799_421170)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_420800' (line 97)
        tuple_var_assignment_420800_421171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420800')
        # Assigning a type to the variable 'itn' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'itn', tuple_var_assignment_420800_421171)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_420801' (line 97)
        tuple_var_assignment_420801_421172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420801')
        # Assigning a type to the variable 'normr' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'normr', tuple_var_assignment_420801_421172)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_420802' (line 97)
        tuple_var_assignment_420802_421173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420802')
        # Assigning a type to the variable 'normar' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 'normar', tuple_var_assignment_420802_421173)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_420803' (line 97)
        tuple_var_assignment_420803_421174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420803')
        # Assigning a type to the variable 'normA' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'normA', tuple_var_assignment_420803_421174)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_420804' (line 97)
        tuple_var_assignment_420804_421175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420804')
        # Assigning a type to the variable 'condA' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 45), 'condA', tuple_var_assignment_420804_421175)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_420805' (line 97)
        tuple_var_assignment_420805_421176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_420805')
        # Assigning a type to the variable 'normx' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 52), 'normx', tuple_var_assignment_420805_421176)
        
        # Call to assert_almost_equal(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'normr' (line 98)
        normr_421178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'normr', False)
        
        # Call to norm(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'self' (line 98)
        self_421180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'self', False)
        # Obtaining the member 'b' of a type (line 98)
        b_421181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 40), self_421180, 'b')
        
        # Call to matvec(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x' (line 98)
        x_421185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 66), 'x', False)
        # Processing the call keyword arguments (line 98)
        kwargs_421186 = {}
        # Getting the type of 'self' (line 98)
        self_421182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 49), 'self', False)
        # Obtaining the member 'Afun' of a type (line 98)
        Afun_421183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 49), self_421182, 'Afun')
        # Obtaining the member 'matvec' of a type (line 98)
        matvec_421184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 49), Afun_421183, 'matvec')
        # Calling matvec(args, kwargs) (line 98)
        matvec_call_result_421187 = invoke(stypy.reporting.localization.Localization(__file__, 98, 49), matvec_421184, *[x_421185], **kwargs_421186)
        
        # Applying the binary operator '-' (line 98)
        result_sub_421188 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 40), '-', b_421181, matvec_call_result_421187)
        
        # Processing the call keyword arguments (line 98)
        kwargs_421189 = {}
        # Getting the type of 'norm' (line 98)
        norm_421179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'norm', False)
        # Calling norm(args, kwargs) (line 98)
        norm_call_result_421190 = invoke(stypy.reporting.localization.Localization(__file__, 98, 35), norm_421179, *[result_sub_421188], **kwargs_421189)
        
        # Processing the call keyword arguments (line 98)
        kwargs_421191 = {}
        # Getting the type of 'assert_almost_equal' (line 98)
        assert_almost_equal_421177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 98)
        assert_almost_equal_call_result_421192 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), assert_almost_equal_421177, *[normr_421178, norm_call_result_421190], **kwargs_421191)
        
        
        # ################# End of 'testNormr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testNormr' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_421193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_421193)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testNormr'
        return stypy_return_type_421193


    @norecursion
    def testNormar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testNormar'
        module_type_store = module_type_store.open_function_context('testNormar', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_localization', localization)
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_function_name', 'TestLSMRReturns.testNormar')
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMRReturns.testNormar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMRReturns.testNormar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testNormar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testNormar(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 101):
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_421194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'self' (line 101)
        self_421195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 101)
        returnValues_421196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), self_421195, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___421197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), returnValues_421196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_421198 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___421197, int_421194)
        
        # Assigning a type to the variable 'tuple_var_assignment_420806' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420806', subscript_call_result_421198)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_421199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'self' (line 101)
        self_421200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 101)
        returnValues_421201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), self_421200, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___421202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), returnValues_421201, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_421203 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___421202, int_421199)
        
        # Assigning a type to the variable 'tuple_var_assignment_420807' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420807', subscript_call_result_421203)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_421204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'self' (line 101)
        self_421205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 101)
        returnValues_421206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), self_421205, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___421207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), returnValues_421206, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_421208 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___421207, int_421204)
        
        # Assigning a type to the variable 'tuple_var_assignment_420808' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420808', subscript_call_result_421208)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_421209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'self' (line 101)
        self_421210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 101)
        returnValues_421211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), self_421210, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___421212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), returnValues_421211, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_421213 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___421212, int_421209)
        
        # Assigning a type to the variable 'tuple_var_assignment_420809' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420809', subscript_call_result_421213)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_421214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'self' (line 101)
        self_421215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 101)
        returnValues_421216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), self_421215, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___421217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), returnValues_421216, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_421218 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___421217, int_421214)
        
        # Assigning a type to the variable 'tuple_var_assignment_420810' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420810', subscript_call_result_421218)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_421219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'self' (line 101)
        self_421220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 101)
        returnValues_421221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), self_421220, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___421222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), returnValues_421221, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_421223 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___421222, int_421219)
        
        # Assigning a type to the variable 'tuple_var_assignment_420811' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420811', subscript_call_result_421223)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_421224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'self' (line 101)
        self_421225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 101)
        returnValues_421226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), self_421225, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___421227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), returnValues_421226, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_421228 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___421227, int_421224)
        
        # Assigning a type to the variable 'tuple_var_assignment_420812' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420812', subscript_call_result_421228)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_421229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'self' (line 101)
        self_421230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 101)
        returnValues_421231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), self_421230, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___421232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), returnValues_421231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_421233 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___421232, int_421229)
        
        # Assigning a type to the variable 'tuple_var_assignment_420813' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420813', subscript_call_result_421233)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_420806' (line 101)
        tuple_var_assignment_420806_421234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420806')
        # Assigning a type to the variable 'x' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'x', tuple_var_assignment_420806_421234)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_420807' (line 101)
        tuple_var_assignment_420807_421235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420807')
        # Assigning a type to the variable 'istop' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'istop', tuple_var_assignment_420807_421235)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_420808' (line 101)
        tuple_var_assignment_420808_421236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420808')
        # Assigning a type to the variable 'itn' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'itn', tuple_var_assignment_420808_421236)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_420809' (line 101)
        tuple_var_assignment_420809_421237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420809')
        # Assigning a type to the variable 'normr' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'normr', tuple_var_assignment_420809_421237)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_420810' (line 101)
        tuple_var_assignment_420810_421238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420810')
        # Assigning a type to the variable 'normar' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'normar', tuple_var_assignment_420810_421238)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_420811' (line 101)
        tuple_var_assignment_420811_421239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420811')
        # Assigning a type to the variable 'normA' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'normA', tuple_var_assignment_420811_421239)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_420812' (line 101)
        tuple_var_assignment_420812_421240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420812')
        # Assigning a type to the variable 'condA' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 45), 'condA', tuple_var_assignment_420812_421240)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_420813' (line 101)
        tuple_var_assignment_420813_421241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_420813')
        # Assigning a type to the variable 'normx' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 52), 'normx', tuple_var_assignment_420813_421241)
        
        # Call to assert_almost_equal(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'normar' (line 102)
        normar_421243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'normar', False)
        
        # Call to norm(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to rmatvec(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_421248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'self', False)
        # Obtaining the member 'b' of a type (line 103)
        b_421249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 39), self_421248, 'b')
        
        # Call to matvec(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'x' (line 103)
        x_421253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 65), 'x', False)
        # Processing the call keyword arguments (line 103)
        kwargs_421254 = {}
        # Getting the type of 'self' (line 103)
        self_421250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 48), 'self', False)
        # Obtaining the member 'Afun' of a type (line 103)
        Afun_421251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 48), self_421250, 'Afun')
        # Obtaining the member 'matvec' of a type (line 103)
        matvec_421252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 48), Afun_421251, 'matvec')
        # Calling matvec(args, kwargs) (line 103)
        matvec_call_result_421255 = invoke(stypy.reporting.localization.Localization(__file__, 103, 48), matvec_421252, *[x_421253], **kwargs_421254)
        
        # Applying the binary operator '-' (line 103)
        result_sub_421256 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 39), '-', b_421249, matvec_call_result_421255)
        
        # Processing the call keyword arguments (line 103)
        kwargs_421257 = {}
        # Getting the type of 'self' (line 103)
        self_421245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 21), 'self', False)
        # Obtaining the member 'Afun' of a type (line 103)
        Afun_421246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 21), self_421245, 'Afun')
        # Obtaining the member 'rmatvec' of a type (line 103)
        rmatvec_421247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 21), Afun_421246, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 103)
        rmatvec_call_result_421258 = invoke(stypy.reporting.localization.Localization(__file__, 103, 21), rmatvec_421247, *[result_sub_421256], **kwargs_421257)
        
        # Processing the call keyword arguments (line 103)
        kwargs_421259 = {}
        # Getting the type of 'norm' (line 103)
        norm_421244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'norm', False)
        # Calling norm(args, kwargs) (line 103)
        norm_call_result_421260 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), norm_421244, *[rmatvec_call_result_421258], **kwargs_421259)
        
        # Processing the call keyword arguments (line 102)
        kwargs_421261 = {}
        # Getting the type of 'assert_almost_equal' (line 102)
        assert_almost_equal_421242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 102)
        assert_almost_equal_call_result_421262 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_almost_equal_421242, *[normar_421243, norm_call_result_421260], **kwargs_421261)
        
        
        # ################# End of 'testNormar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testNormar' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_421263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_421263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testNormar'
        return stypy_return_type_421263


    @norecursion
    def testNormx(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testNormx'
        module_type_store = module_type_store.open_function_context('testNormx', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_localization', localization)
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_function_name', 'TestLSMRReturns.testNormx')
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_param_names_list', [])
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLSMRReturns.testNormx.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMRReturns.testNormx', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testNormx', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testNormx(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 106):
        
        # Assigning a Subscript to a Name (line 106):
        
        # Obtaining the type of the subscript
        int_421264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'self' (line 106)
        self_421265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 106)
        returnValues_421266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), self_421265, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___421267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), returnValues_421266, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_421268 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___421267, int_421264)
        
        # Assigning a type to the variable 'tuple_var_assignment_420814' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420814', subscript_call_result_421268)
        
        # Assigning a Subscript to a Name (line 106):
        
        # Obtaining the type of the subscript
        int_421269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'self' (line 106)
        self_421270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 106)
        returnValues_421271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), self_421270, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___421272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), returnValues_421271, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_421273 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___421272, int_421269)
        
        # Assigning a type to the variable 'tuple_var_assignment_420815' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420815', subscript_call_result_421273)
        
        # Assigning a Subscript to a Name (line 106):
        
        # Obtaining the type of the subscript
        int_421274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'self' (line 106)
        self_421275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 106)
        returnValues_421276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), self_421275, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___421277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), returnValues_421276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_421278 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___421277, int_421274)
        
        # Assigning a type to the variable 'tuple_var_assignment_420816' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420816', subscript_call_result_421278)
        
        # Assigning a Subscript to a Name (line 106):
        
        # Obtaining the type of the subscript
        int_421279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'self' (line 106)
        self_421280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 106)
        returnValues_421281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), self_421280, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___421282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), returnValues_421281, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_421283 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___421282, int_421279)
        
        # Assigning a type to the variable 'tuple_var_assignment_420817' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420817', subscript_call_result_421283)
        
        # Assigning a Subscript to a Name (line 106):
        
        # Obtaining the type of the subscript
        int_421284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'self' (line 106)
        self_421285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 106)
        returnValues_421286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), self_421285, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___421287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), returnValues_421286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_421288 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___421287, int_421284)
        
        # Assigning a type to the variable 'tuple_var_assignment_420818' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420818', subscript_call_result_421288)
        
        # Assigning a Subscript to a Name (line 106):
        
        # Obtaining the type of the subscript
        int_421289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'self' (line 106)
        self_421290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 106)
        returnValues_421291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), self_421290, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___421292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), returnValues_421291, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_421293 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___421292, int_421289)
        
        # Assigning a type to the variable 'tuple_var_assignment_420819' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420819', subscript_call_result_421293)
        
        # Assigning a Subscript to a Name (line 106):
        
        # Obtaining the type of the subscript
        int_421294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'self' (line 106)
        self_421295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 106)
        returnValues_421296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), self_421295, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___421297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), returnValues_421296, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_421298 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___421297, int_421294)
        
        # Assigning a type to the variable 'tuple_var_assignment_420820' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420820', subscript_call_result_421298)
        
        # Assigning a Subscript to a Name (line 106):
        
        # Obtaining the type of the subscript
        int_421299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'self' (line 106)
        self_421300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'self')
        # Obtaining the member 'returnValues' of a type (line 106)
        returnValues_421301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), self_421300, 'returnValues')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___421302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), returnValues_421301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_421303 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___421302, int_421299)
        
        # Assigning a type to the variable 'tuple_var_assignment_420821' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420821', subscript_call_result_421303)
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'tuple_var_assignment_420814' (line 106)
        tuple_var_assignment_420814_421304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420814')
        # Assigning a type to the variable 'x' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'x', tuple_var_assignment_420814_421304)
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'tuple_var_assignment_420815' (line 106)
        tuple_var_assignment_420815_421305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420815')
        # Assigning a type to the variable 'istop' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'istop', tuple_var_assignment_420815_421305)
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'tuple_var_assignment_420816' (line 106)
        tuple_var_assignment_420816_421306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420816')
        # Assigning a type to the variable 'itn' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'itn', tuple_var_assignment_420816_421306)
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'tuple_var_assignment_420817' (line 106)
        tuple_var_assignment_420817_421307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420817')
        # Assigning a type to the variable 'normr' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'normr', tuple_var_assignment_420817_421307)
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'tuple_var_assignment_420818' (line 106)
        tuple_var_assignment_420818_421308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420818')
        # Assigning a type to the variable 'normar' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'normar', tuple_var_assignment_420818_421308)
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'tuple_var_assignment_420819' (line 106)
        tuple_var_assignment_420819_421309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420819')
        # Assigning a type to the variable 'normA' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 38), 'normA', tuple_var_assignment_420819_421309)
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'tuple_var_assignment_420820' (line 106)
        tuple_var_assignment_420820_421310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420820')
        # Assigning a type to the variable 'condA' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 45), 'condA', tuple_var_assignment_420820_421310)
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'tuple_var_assignment_420821' (line 106)
        tuple_var_assignment_420821_421311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_420821')
        # Assigning a type to the variable 'normx' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 52), 'normx', tuple_var_assignment_420821_421311)
        
        # Call to assert_almost_equal(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'normx' (line 107)
        normx_421313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'normx', False)
        
        # Call to norm(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'x' (line 107)
        x_421315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 40), 'x', False)
        # Processing the call keyword arguments (line 107)
        kwargs_421316 = {}
        # Getting the type of 'norm' (line 107)
        norm_421314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 35), 'norm', False)
        # Calling norm(args, kwargs) (line 107)
        norm_call_result_421317 = invoke(stypy.reporting.localization.Localization(__file__, 107, 35), norm_421314, *[x_421315], **kwargs_421316)
        
        # Processing the call keyword arguments (line 107)
        kwargs_421318 = {}
        # Getting the type of 'assert_almost_equal' (line 107)
        assert_almost_equal_421312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 107)
        assert_almost_equal_call_result_421319 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert_almost_equal_421312, *[normx_421313, norm_call_result_421317], **kwargs_421318)
        
        
        # ################# End of 'testNormx(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testNormx' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_421320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_421320)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testNormx'
        return stypy_return_type_421320


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 87, 0, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLSMRReturns.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLSMRReturns' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'TestLSMRReturns', TestLSMRReturns)

@norecursion
def lowerBidiagonalMatrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lowerBidiagonalMatrix'
    module_type_store = module_type_store.open_function_context('lowerBidiagonalMatrix', 110, 0, False)
    
    # Passed parameters checking function
    lowerBidiagonalMatrix.stypy_localization = localization
    lowerBidiagonalMatrix.stypy_type_of_self = None
    lowerBidiagonalMatrix.stypy_type_store = module_type_store
    lowerBidiagonalMatrix.stypy_function_name = 'lowerBidiagonalMatrix'
    lowerBidiagonalMatrix.stypy_param_names_list = ['m', 'n']
    lowerBidiagonalMatrix.stypy_varargs_param_name = None
    lowerBidiagonalMatrix.stypy_kwargs_param_name = None
    lowerBidiagonalMatrix.stypy_call_defaults = defaults
    lowerBidiagonalMatrix.stypy_call_varargs = varargs
    lowerBidiagonalMatrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lowerBidiagonalMatrix', ['m', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lowerBidiagonalMatrix', localization, ['m', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lowerBidiagonalMatrix(...)' code ##################

    
    
    # Getting the type of 'm' (line 122)
    m_421321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 7), 'm')
    # Getting the type of 'n' (line 122)
    n_421322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'n')
    # Applying the binary operator '<=' (line 122)
    result_le_421323 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 7), '<=', m_421321, n_421322)
    
    # Testing the type of an if condition (line 122)
    if_condition_421324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 4), result_le_421323)
    # Assigning a type to the variable 'if_condition_421324' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'if_condition_421324', if_condition_421324)
    # SSA begins for if statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to hstack(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Obtaining an instance of the builtin type 'tuple' (line 123)
    tuple_421326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 123)
    # Adding element type (line 123)
    
    # Call to arange(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'm' (line 123)
    m_421328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'm', False)
    # Processing the call keyword arguments (line 123)
    # Getting the type of 'int' (line 123)
    int_421329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'int', False)
    keyword_421330 = int_421329
    kwargs_421331 = {'dtype': keyword_421330}
    # Getting the type of 'arange' (line 123)
    arange_421327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'arange', False)
    # Calling arange(args, kwargs) (line 123)
    arange_call_result_421332 = invoke(stypy.reporting.localization.Localization(__file__, 123, 22), arange_421327, *[m_421328], **kwargs_421331)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 22), tuple_421326, arange_call_result_421332)
    # Adding element type (line 123)
    
    # Call to arange(...): (line 124)
    # Processing the call arguments (line 124)
    int_421334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 29), 'int')
    # Getting the type of 'm' (line 124)
    m_421335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 32), 'm', False)
    # Processing the call keyword arguments (line 124)
    # Getting the type of 'int' (line 124)
    int_421336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 41), 'int', False)
    keyword_421337 = int_421336
    kwargs_421338 = {'dtype': keyword_421337}
    # Getting the type of 'arange' (line 124)
    arange_421333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 22), 'arange', False)
    # Calling arange(args, kwargs) (line 124)
    arange_call_result_421339 = invoke(stypy.reporting.localization.Localization(__file__, 124, 22), arange_421333, *[int_421334, m_421335], **kwargs_421338)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 22), tuple_421326, arange_call_result_421339)
    
    # Processing the call keyword arguments (line 123)
    kwargs_421340 = {}
    # Getting the type of 'hstack' (line 123)
    hstack_421325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'hstack', False)
    # Calling hstack(args, kwargs) (line 123)
    hstack_call_result_421341 = invoke(stypy.reporting.localization.Localization(__file__, 123, 14), hstack_421325, *[tuple_421326], **kwargs_421340)
    
    # Assigning a type to the variable 'row' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'row', hstack_call_result_421341)
    
    # Assigning a Call to a Name (line 125):
    
    # Assigning a Call to a Name (line 125):
    
    # Call to hstack(...): (line 125)
    # Processing the call arguments (line 125)
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_421343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    
    # Call to arange(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'm' (line 125)
    m_421345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'm', False)
    # Processing the call keyword arguments (line 125)
    # Getting the type of 'int' (line 125)
    int_421346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'int', False)
    keyword_421347 = int_421346
    kwargs_421348 = {'dtype': keyword_421347}
    # Getting the type of 'arange' (line 125)
    arange_421344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'arange', False)
    # Calling arange(args, kwargs) (line 125)
    arange_call_result_421349 = invoke(stypy.reporting.localization.Localization(__file__, 125, 22), arange_421344, *[m_421345], **kwargs_421348)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 22), tuple_421343, arange_call_result_421349)
    # Adding element type (line 125)
    
    # Call to arange(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'm' (line 126)
    m_421351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 29), 'm', False)
    int_421352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 31), 'int')
    # Applying the binary operator '-' (line 126)
    result_sub_421353 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 29), '-', m_421351, int_421352)
    
    # Processing the call keyword arguments (line 126)
    # Getting the type of 'int' (line 126)
    int_421354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 40), 'int', False)
    keyword_421355 = int_421354
    kwargs_421356 = {'dtype': keyword_421355}
    # Getting the type of 'arange' (line 126)
    arange_421350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'arange', False)
    # Calling arange(args, kwargs) (line 126)
    arange_call_result_421357 = invoke(stypy.reporting.localization.Localization(__file__, 126, 22), arange_421350, *[result_sub_421353], **kwargs_421356)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 22), tuple_421343, arange_call_result_421357)
    
    # Processing the call keyword arguments (line 125)
    kwargs_421358 = {}
    # Getting the type of 'hstack' (line 125)
    hstack_421342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 14), 'hstack', False)
    # Calling hstack(args, kwargs) (line 125)
    hstack_call_result_421359 = invoke(stypy.reporting.localization.Localization(__file__, 125, 14), hstack_421342, *[tuple_421343], **kwargs_421358)
    
    # Assigning a type to the variable 'col' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'col', hstack_call_result_421359)
    
    # Assigning a Call to a Name (line 127):
    
    # Assigning a Call to a Name (line 127):
    
    # Call to hstack(...): (line 127)
    # Processing the call arguments (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 127)
    tuple_421361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 127)
    # Adding element type (line 127)
    
    # Call to arange(...): (line 127)
    # Processing the call arguments (line 127)
    int_421363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 30), 'int')
    # Getting the type of 'm' (line 127)
    m_421364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'm', False)
    int_421365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 35), 'int')
    # Applying the binary operator '+' (line 127)
    result_add_421366 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 33), '+', m_421364, int_421365)
    
    # Processing the call keyword arguments (line 127)
    # Getting the type of 'float' (line 127)
    float_421367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 44), 'float', False)
    keyword_421368 = float_421367
    kwargs_421369 = {'dtype': keyword_421368}
    # Getting the type of 'arange' (line 127)
    arange_421362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'arange', False)
    # Calling arange(args, kwargs) (line 127)
    arange_call_result_421370 = invoke(stypy.reporting.localization.Localization(__file__, 127, 23), arange_421362, *[int_421363, result_add_421366], **kwargs_421369)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 23), tuple_421361, arange_call_result_421370)
    # Adding element type (line 127)
    
    # Call to arange(...): (line 128)
    # Processing the call arguments (line 128)
    int_421372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'int')
    # Getting the type of 'm' (line 128)
    m_421373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'm', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'float' (line 128)
    float_421374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'float', False)
    keyword_421375 = float_421374
    kwargs_421376 = {'dtype': keyword_421375}
    # Getting the type of 'arange' (line 128)
    arange_421371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'arange', False)
    # Calling arange(args, kwargs) (line 128)
    arange_call_result_421377 = invoke(stypy.reporting.localization.Localization(__file__, 128, 23), arange_421371, *[int_421372, m_421373], **kwargs_421376)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 23), tuple_421361, arange_call_result_421377)
    
    # Processing the call keyword arguments (line 127)
    kwargs_421378 = {}
    # Getting the type of 'hstack' (line 127)
    hstack_421360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'hstack', False)
    # Calling hstack(args, kwargs) (line 127)
    hstack_call_result_421379 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), hstack_421360, *[tuple_421361], **kwargs_421378)
    
    # Assigning a type to the variable 'data' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'data', hstack_call_result_421379)
    
    # Call to coo_matrix(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Obtaining an instance of the builtin type 'tuple' (line 129)
    tuple_421381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 129)
    # Adding element type (line 129)
    # Getting the type of 'data' (line 129)
    data_421382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 27), tuple_421381, data_421382)
    # Adding element type (line 129)
    
    # Obtaining an instance of the builtin type 'tuple' (line 129)
    tuple_421383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 129)
    # Adding element type (line 129)
    # Getting the type of 'row' (line 129)
    row_421384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 34), 'row', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 34), tuple_421383, row_421384)
    # Adding element type (line 129)
    # Getting the type of 'col' (line 129)
    col_421385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 'col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 34), tuple_421383, col_421385)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 27), tuple_421381, tuple_421383)
    
    # Processing the call keyword arguments (line 129)
    
    # Obtaining an instance of the builtin type 'tuple' (line 129)
    tuple_421386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 129)
    # Adding element type (line 129)
    # Getting the type of 'm' (line 129)
    m_421387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 53), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 53), tuple_421386, m_421387)
    # Adding element type (line 129)
    # Getting the type of 'n' (line 129)
    n_421388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 55), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 53), tuple_421386, n_421388)
    
    keyword_421389 = tuple_421386
    kwargs_421390 = {'shape': keyword_421389}
    # Getting the type of 'coo_matrix' (line 129)
    coo_matrix_421380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 129)
    coo_matrix_call_result_421391 = invoke(stypy.reporting.localization.Localization(__file__, 129, 15), coo_matrix_421380, *[tuple_421381], **kwargs_421390)
    
    # Assigning a type to the variable 'stypy_return_type' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', coo_matrix_call_result_421391)
    # SSA branch for the else part of an if statement (line 122)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to hstack(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Obtaining an instance of the builtin type 'tuple' (line 131)
    tuple_421393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 131)
    # Adding element type (line 131)
    
    # Call to arange(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'n' (line 131)
    n_421395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'n', False)
    # Processing the call keyword arguments (line 131)
    # Getting the type of 'int' (line 131)
    int_421396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 38), 'int', False)
    keyword_421397 = int_421396
    kwargs_421398 = {'dtype': keyword_421397}
    # Getting the type of 'arange' (line 131)
    arange_421394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'arange', False)
    # Calling arange(args, kwargs) (line 131)
    arange_call_result_421399 = invoke(stypy.reporting.localization.Localization(__file__, 131, 22), arange_421394, *[n_421395], **kwargs_421398)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 22), tuple_421393, arange_call_result_421399)
    # Adding element type (line 131)
    
    # Call to arange(...): (line 132)
    # Processing the call arguments (line 132)
    int_421401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 29), 'int')
    # Getting the type of 'n' (line 132)
    n_421402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 32), 'n', False)
    int_421403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 34), 'int')
    # Applying the binary operator '+' (line 132)
    result_add_421404 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 32), '+', n_421402, int_421403)
    
    # Processing the call keyword arguments (line 132)
    # Getting the type of 'int' (line 132)
    int_421405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'int', False)
    keyword_421406 = int_421405
    kwargs_421407 = {'dtype': keyword_421406}
    # Getting the type of 'arange' (line 132)
    arange_421400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 22), 'arange', False)
    # Calling arange(args, kwargs) (line 132)
    arange_call_result_421408 = invoke(stypy.reporting.localization.Localization(__file__, 132, 22), arange_421400, *[int_421401, result_add_421404], **kwargs_421407)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 22), tuple_421393, arange_call_result_421408)
    
    # Processing the call keyword arguments (line 131)
    kwargs_421409 = {}
    # Getting the type of 'hstack' (line 131)
    hstack_421392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), 'hstack', False)
    # Calling hstack(args, kwargs) (line 131)
    hstack_call_result_421410 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), hstack_421392, *[tuple_421393], **kwargs_421409)
    
    # Assigning a type to the variable 'row' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'row', hstack_call_result_421410)
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to hstack(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_421412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    # Adding element type (line 133)
    
    # Call to arange(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'n' (line 133)
    n_421414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'n', False)
    # Processing the call keyword arguments (line 133)
    # Getting the type of 'int' (line 133)
    int_421415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), 'int', False)
    keyword_421416 = int_421415
    kwargs_421417 = {'dtype': keyword_421416}
    # Getting the type of 'arange' (line 133)
    arange_421413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'arange', False)
    # Calling arange(args, kwargs) (line 133)
    arange_call_result_421418 = invoke(stypy.reporting.localization.Localization(__file__, 133, 22), arange_421413, *[n_421414], **kwargs_421417)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), tuple_421412, arange_call_result_421418)
    # Adding element type (line 133)
    
    # Call to arange(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'n' (line 134)
    n_421420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'n', False)
    # Processing the call keyword arguments (line 134)
    # Getting the type of 'int' (line 134)
    int_421421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 38), 'int', False)
    keyword_421422 = int_421421
    kwargs_421423 = {'dtype': keyword_421422}
    # Getting the type of 'arange' (line 134)
    arange_421419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'arange', False)
    # Calling arange(args, kwargs) (line 134)
    arange_call_result_421424 = invoke(stypy.reporting.localization.Localization(__file__, 134, 22), arange_421419, *[n_421420], **kwargs_421423)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), tuple_421412, arange_call_result_421424)
    
    # Processing the call keyword arguments (line 133)
    kwargs_421425 = {}
    # Getting the type of 'hstack' (line 133)
    hstack_421411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 14), 'hstack', False)
    # Calling hstack(args, kwargs) (line 133)
    hstack_call_result_421426 = invoke(stypy.reporting.localization.Localization(__file__, 133, 14), hstack_421411, *[tuple_421412], **kwargs_421425)
    
    # Assigning a type to the variable 'col' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'col', hstack_call_result_421426)
    
    # Assigning a Call to a Name (line 135):
    
    # Assigning a Call to a Name (line 135):
    
    # Call to hstack(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Obtaining an instance of the builtin type 'tuple' (line 135)
    tuple_421428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 135)
    # Adding element type (line 135)
    
    # Call to arange(...): (line 135)
    # Processing the call arguments (line 135)
    int_421430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 30), 'int')
    # Getting the type of 'n' (line 135)
    n_421431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'n', False)
    int_421432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'int')
    # Applying the binary operator '+' (line 135)
    result_add_421433 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 33), '+', n_421431, int_421432)
    
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'float' (line 135)
    float_421434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 44), 'float', False)
    keyword_421435 = float_421434
    kwargs_421436 = {'dtype': keyword_421435}
    # Getting the type of 'arange' (line 135)
    arange_421429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'arange', False)
    # Calling arange(args, kwargs) (line 135)
    arange_call_result_421437 = invoke(stypy.reporting.localization.Localization(__file__, 135, 23), arange_421429, *[int_421430, result_add_421433], **kwargs_421436)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 23), tuple_421428, arange_call_result_421437)
    # Adding element type (line 135)
    
    # Call to arange(...): (line 136)
    # Processing the call arguments (line 136)
    int_421439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 30), 'int')
    # Getting the type of 'n' (line 136)
    n_421440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 32), 'n', False)
    int_421441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 34), 'int')
    # Applying the binary operator '+' (line 136)
    result_add_421442 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 32), '+', n_421440, int_421441)
    
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'float' (line 136)
    float_421443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 43), 'float', False)
    keyword_421444 = float_421443
    kwargs_421445 = {'dtype': keyword_421444}
    # Getting the type of 'arange' (line 136)
    arange_421438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'arange', False)
    # Calling arange(args, kwargs) (line 136)
    arange_call_result_421446 = invoke(stypy.reporting.localization.Localization(__file__, 136, 23), arange_421438, *[int_421439, result_add_421442], **kwargs_421445)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 23), tuple_421428, arange_call_result_421446)
    
    # Processing the call keyword arguments (line 135)
    kwargs_421447 = {}
    # Getting the type of 'hstack' (line 135)
    hstack_421427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'hstack', False)
    # Calling hstack(args, kwargs) (line 135)
    hstack_call_result_421448 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), hstack_421427, *[tuple_421428], **kwargs_421447)
    
    # Assigning a type to the variable 'data' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'data', hstack_call_result_421448)
    
    # Call to coo_matrix(...): (line 137)
    # Processing the call arguments (line 137)
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_421450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'data' (line 137)
    data_421451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 27), tuple_421450, data_421451)
    # Adding element type (line 137)
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_421452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'row' (line 137)
    row_421453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'row', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 33), tuple_421452, row_421453)
    # Adding element type (line 137)
    # Getting the type of 'col' (line 137)
    col_421454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 33), tuple_421452, col_421454)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 27), tuple_421450, tuple_421452)
    
    # Processing the call keyword arguments (line 137)
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_421455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'm' (line 137)
    m_421456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 52), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 52), tuple_421455, m_421456)
    # Adding element type (line 137)
    # Getting the type of 'n' (line 137)
    n_421457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 54), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 52), tuple_421455, n_421457)
    
    keyword_421458 = tuple_421455
    kwargs_421459 = {'shape': keyword_421458}
    # Getting the type of 'coo_matrix' (line 137)
    coo_matrix_421449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 137)
    coo_matrix_call_result_421460 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), coo_matrix_421449, *[tuple_421450], **kwargs_421459)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', coo_matrix_call_result_421460)
    # SSA join for if statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lowerBidiagonalMatrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lowerBidiagonalMatrix' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_421461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421461)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lowerBidiagonalMatrix'
    return stypy_return_type_421461

# Assigning a type to the variable 'lowerBidiagonalMatrix' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'lowerBidiagonalMatrix', lowerBidiagonalMatrix)

@norecursion
def lsmrtest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lsmrtest'
    module_type_store = module_type_store.open_function_context('lsmrtest', 140, 0, False)
    
    # Passed parameters checking function
    lsmrtest.stypy_localization = localization
    lsmrtest.stypy_type_of_self = None
    lsmrtest.stypy_type_store = module_type_store
    lsmrtest.stypy_function_name = 'lsmrtest'
    lsmrtest.stypy_param_names_list = ['m', 'n', 'damp']
    lsmrtest.stypy_varargs_param_name = None
    lsmrtest.stypy_kwargs_param_name = None
    lsmrtest.stypy_call_defaults = defaults
    lsmrtest.stypy_call_varargs = varargs
    lsmrtest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lsmrtest', ['m', 'n', 'damp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lsmrtest', localization, ['m', 'n', 'damp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lsmrtest(...)' code ##################

    str_421462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 4), 'str', 'Verbose testing of lsmr')
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to lowerBidiagonalMatrix(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'm' (line 143)
    m_421464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'm', False)
    # Getting the type of 'n' (line 143)
    n_421465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 32), 'n', False)
    # Processing the call keyword arguments (line 143)
    kwargs_421466 = {}
    # Getting the type of 'lowerBidiagonalMatrix' (line 143)
    lowerBidiagonalMatrix_421463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'lowerBidiagonalMatrix', False)
    # Calling lowerBidiagonalMatrix(args, kwargs) (line 143)
    lowerBidiagonalMatrix_call_result_421467 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), lowerBidiagonalMatrix_421463, *[m_421464, n_421465], **kwargs_421466)
    
    # Assigning a type to the variable 'A' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'A', lowerBidiagonalMatrix_call_result_421467)
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to arange(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'n' (line 144)
    n_421469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'n', False)
    int_421470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'int')
    int_421471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'int')
    # Processing the call keyword arguments (line 144)
    # Getting the type of 'float' (line 144)
    float_421472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'float', False)
    keyword_421473 = float_421472
    kwargs_421474 = {'dtype': keyword_421473}
    # Getting the type of 'arange' (line 144)
    arange_421468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'arange', False)
    # Calling arange(args, kwargs) (line 144)
    arange_call_result_421475 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), arange_421468, *[n_421469, int_421470, int_421471], **kwargs_421474)
    
    # Assigning a type to the variable 'xtrue' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'xtrue', arange_call_result_421475)
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to aslinearoperator(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'A' (line 145)
    A_421477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'A', False)
    # Processing the call keyword arguments (line 145)
    kwargs_421478 = {}
    # Getting the type of 'aslinearoperator' (line 145)
    aslinearoperator_421476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 145)
    aslinearoperator_call_result_421479 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), aslinearoperator_421476, *[A_421477], **kwargs_421478)
    
    # Assigning a type to the variable 'Afun' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'Afun', aslinearoperator_call_result_421479)
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to matvec(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'xtrue' (line 147)
    xtrue_421482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'xtrue', False)
    # Processing the call keyword arguments (line 147)
    kwargs_421483 = {}
    # Getting the type of 'Afun' (line 147)
    Afun_421480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'Afun', False)
    # Obtaining the member 'matvec' of a type (line 147)
    matvec_421481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), Afun_421480, 'matvec')
    # Calling matvec(args, kwargs) (line 147)
    matvec_call_result_421484 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), matvec_421481, *[xtrue_421482], **kwargs_421483)
    
    # Assigning a type to the variable 'b' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'b', matvec_call_result_421484)
    
    # Assigning a Num to a Name (line 149):
    
    # Assigning a Num to a Name (line 149):
    float_421485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 11), 'float')
    # Assigning a type to the variable 'atol' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'atol', float_421485)
    
    # Assigning a Num to a Name (line 150):
    
    # Assigning a Num to a Name (line 150):
    float_421486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 11), 'float')
    # Assigning a type to the variable 'btol' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'btol', float_421486)
    
    # Assigning a Num to a Name (line 151):
    
    # Assigning a Num to a Name (line 151):
    float_421487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 13), 'float')
    # Assigning a type to the variable 'conlim' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'conlim', float_421487)
    
    # Assigning a BinOp to a Name (line 152):
    
    # Assigning a BinOp to a Name (line 152):
    int_421488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 13), 'int')
    # Getting the type of 'n' (line 152)
    n_421489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'n')
    # Applying the binary operator '*' (line 152)
    result_mul_421490 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 13), '*', int_421488, n_421489)
    
    # Assigning a type to the variable 'itnlim' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'itnlim', result_mul_421490)
    
    # Assigning a Num to a Name (line 153):
    
    # Assigning a Num to a Name (line 153):
    int_421491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 11), 'int')
    # Assigning a type to the variable 'show' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'show', int_421491)
    
    # Assigning a Call to a Tuple (line 155):
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_421492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to lsmr(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_421494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'A', False)
    # Getting the type of 'b' (line 156)
    b_421495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'b', False)
    # Getting the type of 'damp' (line 156)
    damp_421496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'damp', False)
    # Getting the type of 'atol' (line 156)
    atol_421497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'atol', False)
    # Getting the type of 'btol' (line 156)
    btol_421498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'btol', False)
    # Getting the type of 'conlim' (line 156)
    conlim_421499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'conlim', False)
    # Getting the type of 'itnlim' (line 156)
    itnlim_421500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'itnlim', False)
    # Getting the type of 'show' (line 156)
    show_421501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'show', False)
    # Processing the call keyword arguments (line 156)
    kwargs_421502 = {}
    # Getting the type of 'lsmr' (line 156)
    lsmr_421493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 156)
    lsmr_call_result_421503 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), lsmr_421493, *[A_421494, b_421495, damp_421496, atol_421497, btol_421498, conlim_421499, itnlim_421500, show_421501], **kwargs_421502)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___421504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), lsmr_call_result_421503, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_421505 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___421504, int_421492)
    
    # Assigning a type to the variable 'tuple_var_assignment_420822' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420822', subscript_call_result_421505)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_421506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to lsmr(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_421508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'A', False)
    # Getting the type of 'b' (line 156)
    b_421509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'b', False)
    # Getting the type of 'damp' (line 156)
    damp_421510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'damp', False)
    # Getting the type of 'atol' (line 156)
    atol_421511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'atol', False)
    # Getting the type of 'btol' (line 156)
    btol_421512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'btol', False)
    # Getting the type of 'conlim' (line 156)
    conlim_421513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'conlim', False)
    # Getting the type of 'itnlim' (line 156)
    itnlim_421514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'itnlim', False)
    # Getting the type of 'show' (line 156)
    show_421515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'show', False)
    # Processing the call keyword arguments (line 156)
    kwargs_421516 = {}
    # Getting the type of 'lsmr' (line 156)
    lsmr_421507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 156)
    lsmr_call_result_421517 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), lsmr_421507, *[A_421508, b_421509, damp_421510, atol_421511, btol_421512, conlim_421513, itnlim_421514, show_421515], **kwargs_421516)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___421518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), lsmr_call_result_421517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_421519 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___421518, int_421506)
    
    # Assigning a type to the variable 'tuple_var_assignment_420823' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420823', subscript_call_result_421519)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_421520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to lsmr(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_421522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'A', False)
    # Getting the type of 'b' (line 156)
    b_421523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'b', False)
    # Getting the type of 'damp' (line 156)
    damp_421524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'damp', False)
    # Getting the type of 'atol' (line 156)
    atol_421525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'atol', False)
    # Getting the type of 'btol' (line 156)
    btol_421526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'btol', False)
    # Getting the type of 'conlim' (line 156)
    conlim_421527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'conlim', False)
    # Getting the type of 'itnlim' (line 156)
    itnlim_421528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'itnlim', False)
    # Getting the type of 'show' (line 156)
    show_421529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'show', False)
    # Processing the call keyword arguments (line 156)
    kwargs_421530 = {}
    # Getting the type of 'lsmr' (line 156)
    lsmr_421521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 156)
    lsmr_call_result_421531 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), lsmr_421521, *[A_421522, b_421523, damp_421524, atol_421525, btol_421526, conlim_421527, itnlim_421528, show_421529], **kwargs_421530)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___421532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), lsmr_call_result_421531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_421533 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___421532, int_421520)
    
    # Assigning a type to the variable 'tuple_var_assignment_420824' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420824', subscript_call_result_421533)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_421534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to lsmr(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_421536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'A', False)
    # Getting the type of 'b' (line 156)
    b_421537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'b', False)
    # Getting the type of 'damp' (line 156)
    damp_421538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'damp', False)
    # Getting the type of 'atol' (line 156)
    atol_421539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'atol', False)
    # Getting the type of 'btol' (line 156)
    btol_421540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'btol', False)
    # Getting the type of 'conlim' (line 156)
    conlim_421541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'conlim', False)
    # Getting the type of 'itnlim' (line 156)
    itnlim_421542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'itnlim', False)
    # Getting the type of 'show' (line 156)
    show_421543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'show', False)
    # Processing the call keyword arguments (line 156)
    kwargs_421544 = {}
    # Getting the type of 'lsmr' (line 156)
    lsmr_421535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 156)
    lsmr_call_result_421545 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), lsmr_421535, *[A_421536, b_421537, damp_421538, atol_421539, btol_421540, conlim_421541, itnlim_421542, show_421543], **kwargs_421544)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___421546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), lsmr_call_result_421545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_421547 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___421546, int_421534)
    
    # Assigning a type to the variable 'tuple_var_assignment_420825' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420825', subscript_call_result_421547)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_421548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to lsmr(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_421550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'A', False)
    # Getting the type of 'b' (line 156)
    b_421551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'b', False)
    # Getting the type of 'damp' (line 156)
    damp_421552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'damp', False)
    # Getting the type of 'atol' (line 156)
    atol_421553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'atol', False)
    # Getting the type of 'btol' (line 156)
    btol_421554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'btol', False)
    # Getting the type of 'conlim' (line 156)
    conlim_421555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'conlim', False)
    # Getting the type of 'itnlim' (line 156)
    itnlim_421556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'itnlim', False)
    # Getting the type of 'show' (line 156)
    show_421557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'show', False)
    # Processing the call keyword arguments (line 156)
    kwargs_421558 = {}
    # Getting the type of 'lsmr' (line 156)
    lsmr_421549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 156)
    lsmr_call_result_421559 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), lsmr_421549, *[A_421550, b_421551, damp_421552, atol_421553, btol_421554, conlim_421555, itnlim_421556, show_421557], **kwargs_421558)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___421560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), lsmr_call_result_421559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_421561 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___421560, int_421548)
    
    # Assigning a type to the variable 'tuple_var_assignment_420826' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420826', subscript_call_result_421561)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_421562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to lsmr(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_421564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'A', False)
    # Getting the type of 'b' (line 156)
    b_421565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'b', False)
    # Getting the type of 'damp' (line 156)
    damp_421566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'damp', False)
    # Getting the type of 'atol' (line 156)
    atol_421567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'atol', False)
    # Getting the type of 'btol' (line 156)
    btol_421568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'btol', False)
    # Getting the type of 'conlim' (line 156)
    conlim_421569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'conlim', False)
    # Getting the type of 'itnlim' (line 156)
    itnlim_421570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'itnlim', False)
    # Getting the type of 'show' (line 156)
    show_421571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'show', False)
    # Processing the call keyword arguments (line 156)
    kwargs_421572 = {}
    # Getting the type of 'lsmr' (line 156)
    lsmr_421563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 156)
    lsmr_call_result_421573 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), lsmr_421563, *[A_421564, b_421565, damp_421566, atol_421567, btol_421568, conlim_421569, itnlim_421570, show_421571], **kwargs_421572)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___421574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), lsmr_call_result_421573, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_421575 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___421574, int_421562)
    
    # Assigning a type to the variable 'tuple_var_assignment_420827' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420827', subscript_call_result_421575)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_421576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to lsmr(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_421578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'A', False)
    # Getting the type of 'b' (line 156)
    b_421579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'b', False)
    # Getting the type of 'damp' (line 156)
    damp_421580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'damp', False)
    # Getting the type of 'atol' (line 156)
    atol_421581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'atol', False)
    # Getting the type of 'btol' (line 156)
    btol_421582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'btol', False)
    # Getting the type of 'conlim' (line 156)
    conlim_421583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'conlim', False)
    # Getting the type of 'itnlim' (line 156)
    itnlim_421584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'itnlim', False)
    # Getting the type of 'show' (line 156)
    show_421585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'show', False)
    # Processing the call keyword arguments (line 156)
    kwargs_421586 = {}
    # Getting the type of 'lsmr' (line 156)
    lsmr_421577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 156)
    lsmr_call_result_421587 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), lsmr_421577, *[A_421578, b_421579, damp_421580, atol_421581, btol_421582, conlim_421583, itnlim_421584, show_421585], **kwargs_421586)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___421588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), lsmr_call_result_421587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_421589 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___421588, int_421576)
    
    # Assigning a type to the variable 'tuple_var_assignment_420828' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420828', subscript_call_result_421589)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_421590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')
    
    # Call to lsmr(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_421592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'A', False)
    # Getting the type of 'b' (line 156)
    b_421593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'b', False)
    # Getting the type of 'damp' (line 156)
    damp_421594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'damp', False)
    # Getting the type of 'atol' (line 156)
    atol_421595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'atol', False)
    # Getting the type of 'btol' (line 156)
    btol_421596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'btol', False)
    # Getting the type of 'conlim' (line 156)
    conlim_421597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'conlim', False)
    # Getting the type of 'itnlim' (line 156)
    itnlim_421598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'itnlim', False)
    # Getting the type of 'show' (line 156)
    show_421599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'show', False)
    # Processing the call keyword arguments (line 156)
    kwargs_421600 = {}
    # Getting the type of 'lsmr' (line 156)
    lsmr_421591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 156)
    lsmr_call_result_421601 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), lsmr_421591, *[A_421592, b_421593, damp_421594, atol_421595, btol_421596, conlim_421597, itnlim_421598, show_421599], **kwargs_421600)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___421602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), lsmr_call_result_421601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_421603 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___421602, int_421590)
    
    # Assigning a type to the variable 'tuple_var_assignment_420829' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420829', subscript_call_result_421603)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_420822' (line 155)
    tuple_var_assignment_420822_421604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420822')
    # Assigning a type to the variable 'x' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'x', tuple_var_assignment_420822_421604)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_420823' (line 155)
    tuple_var_assignment_420823_421605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420823')
    # Assigning a type to the variable 'istop' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 7), 'istop', tuple_var_assignment_420823_421605)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_420824' (line 155)
    tuple_var_assignment_420824_421606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420824')
    # Assigning a type to the variable 'itn' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'itn', tuple_var_assignment_420824_421606)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_420825' (line 155)
    tuple_var_assignment_420825_421607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420825')
    # Assigning a type to the variable 'normr' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'normr', tuple_var_assignment_420825_421607)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_420826' (line 155)
    tuple_var_assignment_420826_421608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420826')
    # Assigning a type to the variable 'normar' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'normar', tuple_var_assignment_420826_421608)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_420827' (line 155)
    tuple_var_assignment_420827_421609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420827')
    # Assigning a type to the variable 'norma' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'norma', tuple_var_assignment_420827_421609)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_420828' (line 155)
    tuple_var_assignment_420828_421610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420828')
    # Assigning a type to the variable 'conda' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'conda', tuple_var_assignment_420828_421610)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_420829' (line 155)
    tuple_var_assignment_420829_421611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'tuple_var_assignment_420829')
    # Assigning a type to the variable 'normx' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 48), 'normx', tuple_var_assignment_420829_421611)
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to min(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'n' (line 158)
    n_421613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 13), 'n', False)
    int_421614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 15), 'int')
    # Processing the call keyword arguments (line 158)
    kwargs_421615 = {}
    # Getting the type of 'min' (line 158)
    min_421612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 9), 'min', False)
    # Calling min(args, kwargs) (line 158)
    min_call_result_421616 = invoke(stypy.reporting.localization.Localization(__file__, 158, 9), min_421612, *[n_421613, int_421614], **kwargs_421615)
    
    # Assigning a type to the variable 'j1' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'j1', min_call_result_421616)
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to max(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'n' (line 159)
    n_421618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'n', False)
    int_421619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 15), 'int')
    # Applying the binary operator '-' (line 159)
    result_sub_421620 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 13), '-', n_421618, int_421619)
    
    int_421621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 17), 'int')
    # Processing the call keyword arguments (line 159)
    kwargs_421622 = {}
    # Getting the type of 'max' (line 159)
    max_421617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 9), 'max', False)
    # Calling max(args, kwargs) (line 159)
    max_call_result_421623 = invoke(stypy.reporting.localization.Localization(__file__, 159, 9), max_421617, *[result_sub_421620, int_421621], **kwargs_421622)
    
    # Assigning a type to the variable 'j2' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'j2', max_call_result_421623)
    
    # Call to print(...): (line 160)
    # Processing the call arguments (line 160)
    str_421625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 10), 'str', ' ')
    # Processing the call keyword arguments (line 160)
    kwargs_421626 = {}
    # Getting the type of 'print' (line 160)
    print_421624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'print', False)
    # Calling print(args, kwargs) (line 160)
    print_call_result_421627 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), print_421624, *[str_421625], **kwargs_421626)
    
    
    # Call to print(...): (line 161)
    # Processing the call arguments (line 161)
    str_421629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 10), 'str', 'First elements of x:')
    # Processing the call keyword arguments (line 161)
    kwargs_421630 = {}
    # Getting the type of 'print' (line 161)
    print_421628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'print', False)
    # Calling print(args, kwargs) (line 161)
    print_call_result_421631 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), print_421628, *[str_421629], **kwargs_421630)
    
    
    # Assigning a ListComp to a Name (line 162):
    
    # Assigning a ListComp to a Name (line 162):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_421635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 39), 'int')
    # Getting the type of 'j1' (line 162)
    j1_421636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'j1')
    slice_421637 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 162, 37), int_421635, j1_421636, None)
    # Getting the type of 'x' (line 162)
    x_421638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'x')
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___421639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 37), x_421638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_421640 = invoke(stypy.reporting.localization.Localization(__file__, 162, 37), getitem___421639, slice_421637)
    
    comprehension_421641 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 11), subscript_call_result_421640)
    # Assigning a type to the variable 'xi' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'xi', comprehension_421641)
    str_421632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 11), 'str', '%10.4f')
    # Getting the type of 'xi' (line 162)
    xi_421633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 23), 'xi')
    # Applying the binary operator '%' (line 162)
    result_mod_421634 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), '%', str_421632, xi_421633)
    
    list_421642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 11), list_421642, result_mod_421634)
    # Assigning a type to the variable 'str' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'str', list_421642)
    
    # Call to print(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Call to join(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'str' (line 163)
    str_421646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'str', False)
    # Processing the call keyword arguments (line 163)
    kwargs_421647 = {}
    str_421644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 10), 'str', '')
    # Obtaining the member 'join' of a type (line 163)
    join_421645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 10), str_421644, 'join')
    # Calling join(args, kwargs) (line 163)
    join_call_result_421648 = invoke(stypy.reporting.localization.Localization(__file__, 163, 10), join_421645, *[str_421646], **kwargs_421647)
    
    # Processing the call keyword arguments (line 163)
    kwargs_421649 = {}
    # Getting the type of 'print' (line 163)
    print_421643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'print', False)
    # Calling print(args, kwargs) (line 163)
    print_call_result_421650 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), print_421643, *[join_call_result_421648], **kwargs_421649)
    
    
    # Call to print(...): (line 164)
    # Processing the call arguments (line 164)
    str_421652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 10), 'str', ' ')
    # Processing the call keyword arguments (line 164)
    kwargs_421653 = {}
    # Getting the type of 'print' (line 164)
    print_421651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'print', False)
    # Calling print(args, kwargs) (line 164)
    print_call_result_421654 = invoke(stypy.reporting.localization.Localization(__file__, 164, 4), print_421651, *[str_421652], **kwargs_421653)
    
    
    # Call to print(...): (line 165)
    # Processing the call arguments (line 165)
    str_421656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 10), 'str', 'Last  elements of x:')
    # Processing the call keyword arguments (line 165)
    kwargs_421657 = {}
    # Getting the type of 'print' (line 165)
    print_421655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'print', False)
    # Calling print(args, kwargs) (line 165)
    print_call_result_421658 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), print_421655, *[str_421656], **kwargs_421657)
    
    
    # Assigning a ListComp to a Name (line 166):
    
    # Assigning a ListComp to a Name (line 166):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 'j2' (line 166)
    j2_421662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 39), 'j2')
    int_421663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 42), 'int')
    # Applying the binary operator '-' (line 166)
    result_sub_421664 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 39), '-', j2_421662, int_421663)
    
    slice_421665 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 37), result_sub_421664, None, None)
    # Getting the type of 'x' (line 166)
    x_421666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'x')
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___421667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 37), x_421666, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_421668 = invoke(stypy.reporting.localization.Localization(__file__, 166, 37), getitem___421667, slice_421665)
    
    comprehension_421669 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 11), subscript_call_result_421668)
    # Assigning a type to the variable 'xi' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'xi', comprehension_421669)
    str_421659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 11), 'str', '%10.4f')
    # Getting the type of 'xi' (line 166)
    xi_421660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 23), 'xi')
    # Applying the binary operator '%' (line 166)
    result_mod_421661 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 11), '%', str_421659, xi_421660)
    
    list_421670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 11), list_421670, result_mod_421661)
    # Assigning a type to the variable 'str' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'str', list_421670)
    
    # Call to print(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Call to join(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'str' (line 167)
    str_421674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'str', False)
    # Processing the call keyword arguments (line 167)
    kwargs_421675 = {}
    str_421672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 10), 'str', '')
    # Obtaining the member 'join' of a type (line 167)
    join_421673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 10), str_421672, 'join')
    # Calling join(args, kwargs) (line 167)
    join_call_result_421676 = invoke(stypy.reporting.localization.Localization(__file__, 167, 10), join_421673, *[str_421674], **kwargs_421675)
    
    # Processing the call keyword arguments (line 167)
    kwargs_421677 = {}
    # Getting the type of 'print' (line 167)
    print_421671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'print', False)
    # Calling print(args, kwargs) (line 167)
    print_call_result_421678 = invoke(stypy.reporting.localization.Localization(__file__, 167, 4), print_421671, *[join_call_result_421676], **kwargs_421677)
    
    
    # Assigning a BinOp to a Name (line 169):
    
    # Assigning a BinOp to a Name (line 169):
    # Getting the type of 'b' (line 169)
    b_421679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'b')
    
    # Call to matvec(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'x' (line 169)
    x_421682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'x', False)
    # Processing the call keyword arguments (line 169)
    kwargs_421683 = {}
    # Getting the type of 'Afun' (line 169)
    Afun_421680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'Afun', False)
    # Obtaining the member 'matvec' of a type (line 169)
    matvec_421681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), Afun_421680, 'matvec')
    # Calling matvec(args, kwargs) (line 169)
    matvec_call_result_421684 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), matvec_421681, *[x_421682], **kwargs_421683)
    
    # Applying the binary operator '-' (line 169)
    result_sub_421685 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 8), '-', b_421679, matvec_call_result_421684)
    
    # Assigning a type to the variable 'r' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'r', result_sub_421685)
    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to sqrt(...): (line 170)
    # Processing the call arguments (line 170)
    
    # Call to norm(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'r' (line 170)
    r_421688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'r', False)
    # Processing the call keyword arguments (line 170)
    kwargs_421689 = {}
    # Getting the type of 'norm' (line 170)
    norm_421687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 14), 'norm', False)
    # Calling norm(args, kwargs) (line 170)
    norm_call_result_421690 = invoke(stypy.reporting.localization.Localization(__file__, 170, 14), norm_421687, *[r_421688], **kwargs_421689)
    
    int_421691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'int')
    # Applying the binary operator '**' (line 170)
    result_pow_421692 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 14), '**', norm_call_result_421690, int_421691)
    
    # Getting the type of 'damp' (line 170)
    damp_421693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'damp', False)
    
    # Call to norm(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'x' (line 170)
    x_421695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 38), 'x', False)
    # Processing the call keyword arguments (line 170)
    kwargs_421696 = {}
    # Getting the type of 'norm' (line 170)
    norm_421694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'norm', False)
    # Calling norm(args, kwargs) (line 170)
    norm_call_result_421697 = invoke(stypy.reporting.localization.Localization(__file__, 170, 33), norm_421694, *[x_421695], **kwargs_421696)
    
    # Applying the binary operator '*' (line 170)
    result_mul_421698 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 28), '*', damp_421693, norm_call_result_421697)
    
    int_421699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 43), 'int')
    # Applying the binary operator '**' (line 170)
    result_pow_421700 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 27), '**', result_mul_421698, int_421699)
    
    # Applying the binary operator '+' (line 170)
    result_add_421701 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 14), '+', result_pow_421692, result_pow_421700)
    
    # Processing the call keyword arguments (line 170)
    kwargs_421702 = {}
    # Getting the type of 'sqrt' (line 170)
    sqrt_421686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 9), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 170)
    sqrt_call_result_421703 = invoke(stypy.reporting.localization.Localization(__file__, 170, 9), sqrt_421686, *[result_add_421701], **kwargs_421702)
    
    # Assigning a type to the variable 'r2' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'r2', sqrt_call_result_421703)
    
    # Call to print(...): (line 171)
    # Processing the call arguments (line 171)
    str_421705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 10), 'str', ' ')
    # Processing the call keyword arguments (line 171)
    kwargs_421706 = {}
    # Getting the type of 'print' (line 171)
    print_421704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'print', False)
    # Calling print(args, kwargs) (line 171)
    print_call_result_421707 = invoke(stypy.reporting.localization.Localization(__file__, 171, 4), print_421704, *[str_421705], **kwargs_421706)
    
    
    # Assigning a BinOp to a Name (line 172):
    
    # Assigning a BinOp to a Name (line 172):
    str_421708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 10), 'str', 'normr (est.)  %17.10e')
    # Getting the type of 'normr' (line 172)
    normr_421709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'normr')
    # Applying the binary operator '%' (line 172)
    result_mod_421710 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 10), '%', str_421708, normr_421709)
    
    # Assigning a type to the variable 'str' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'str', result_mod_421710)
    
    # Assigning a BinOp to a Name (line 173):
    
    # Assigning a BinOp to a Name (line 173):
    str_421711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 11), 'str', 'normr (true)  %17.10e')
    # Getting the type of 'r2' (line 173)
    r2_421712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'r2')
    # Applying the binary operator '%' (line 173)
    result_mod_421713 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 11), '%', str_421711, r2_421712)
    
    # Assigning a type to the variable 'str2' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'str2', result_mod_421713)
    
    # Call to print(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'str' (line 174)
    str_421715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 10), 'str', False)
    # Processing the call keyword arguments (line 174)
    kwargs_421716 = {}
    # Getting the type of 'print' (line 174)
    print_421714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'print', False)
    # Calling print(args, kwargs) (line 174)
    print_call_result_421717 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), print_421714, *[str_421715], **kwargs_421716)
    
    
    # Call to print(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'str2' (line 175)
    str2_421719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 10), 'str2', False)
    # Processing the call keyword arguments (line 175)
    kwargs_421720 = {}
    # Getting the type of 'print' (line 175)
    print_421718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'print', False)
    # Calling print(args, kwargs) (line 175)
    print_call_result_421721 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), print_421718, *[str2_421719], **kwargs_421720)
    
    
    # Call to print(...): (line 176)
    # Processing the call arguments (line 176)
    str_421723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 10), 'str', ' ')
    # Processing the call keyword arguments (line 176)
    kwargs_421724 = {}
    # Getting the type of 'print' (line 176)
    print_421722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'print', False)
    # Calling print(args, kwargs) (line 176)
    print_call_result_421725 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), print_421722, *[str_421723], **kwargs_421724)
    
    
    # ################# End of 'lsmrtest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lsmrtest' in the type store
    # Getting the type of 'stypy_return_type' (line 140)
    stypy_return_type_421726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421726)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lsmrtest'
    return stypy_return_type_421726

# Assigning a type to the variable 'lsmrtest' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'lsmrtest', lsmrtest)

if (__name__ == '__main__'):
    
    # Call to lsmrtest(...): (line 179)
    # Processing the call arguments (line 179)
    int_421728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 13), 'int')
    int_421729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'int')
    int_421730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 19), 'int')
    # Processing the call keyword arguments (line 179)
    kwargs_421731 = {}
    # Getting the type of 'lsmrtest' (line 179)
    lsmrtest_421727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'lsmrtest', False)
    # Calling lsmrtest(args, kwargs) (line 179)
    lsmrtest_call_result_421732 = invoke(stypy.reporting.localization.Localization(__file__, 179, 4), lsmrtest_421727, *[int_421728, int_421729, int_421730], **kwargs_421731)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
