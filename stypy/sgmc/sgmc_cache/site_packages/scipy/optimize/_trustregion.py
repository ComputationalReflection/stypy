
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Trust-region optimization.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import math
5: 
6: import numpy as np
7: import scipy.linalg
8: from .optimize import (_check_unknown_options, wrap_function, _status_message,
9:                        OptimizeResult)
10: 
11: __all__ = []
12: 
13: 
14: class BaseQuadraticSubproblem(object):
15:     '''
16:     Base/abstract class defining the quadratic model for trust-region
17:     minimization. Child classes must implement the ``solve`` method.
18: 
19:     Values of the objective function, jacobian and hessian (if provided) at
20:     the current iterate ``x`` are evaluated on demand and then stored as
21:     attributes ``fun``, ``jac``, ``hess``.
22:     '''
23: 
24:     def __init__(self, x, fun, jac, hess=None, hessp=None):
25:         self._x = x
26:         self._f = None
27:         self._g = None
28:         self._h = None
29:         self._g_mag = None
30:         self._cauchy_point = None
31:         self._newton_point = None
32:         self._fun = fun
33:         self._jac = jac
34:         self._hess = hess
35:         self._hessp = hessp
36: 
37:     def __call__(self, p):
38:         return self.fun + np.dot(self.jac, p) + 0.5 * np.dot(p, self.hessp(p))
39: 
40:     @property
41:     def fun(self):
42:         '''Value of objective function at current iteration.'''
43:         if self._f is None:
44:             self._f = self._fun(self._x)
45:         return self._f
46: 
47:     @property
48:     def jac(self):
49:         '''Value of jacobian of objective function at current iteration.'''
50:         if self._g is None:
51:             self._g = self._jac(self._x)
52:         return self._g
53: 
54:     @property
55:     def hess(self):
56:         '''Value of hessian of objective function at current iteration.'''
57:         if self._h is None:
58:             self._h = self._hess(self._x)
59:         return self._h
60: 
61:     def hessp(self, p):
62:         if self._hessp is not None:
63:             return self._hessp(self._x, p)
64:         else:
65:             return np.dot(self.hess, p)
66: 
67:     @property
68:     def jac_mag(self):
69:         '''Magniture of jacobian of objective function at current iteration.'''
70:         if self._g_mag is None:
71:             self._g_mag = scipy.linalg.norm(self.jac)
72:         return self._g_mag
73: 
74:     def get_boundaries_intersections(self, z, d, trust_radius):
75:         '''
76:         Solve the scalar quadratic equation ||z + t d|| == trust_radius.
77:         This is like a line-sphere intersection.
78:         Return the two values of t, sorted from low to high.
79:         '''
80:         a = np.dot(d, d)
81:         b = 2 * np.dot(z, d)
82:         c = np.dot(z, z) - trust_radius**2
83:         sqrt_discriminant = math.sqrt(b*b - 4*a*c)
84: 
85:         # The following calculation is mathematically
86:         # equivalent to:
87:         # ta = (-b - sqrt_discriminant) / (2*a)
88:         # tb = (-b + sqrt_discriminant) / (2*a)
89:         # but produce smaller round off errors.
90:         # Look at Matrix Computation p.97
91:         # for a better justification.
92:         aux = b + math.copysign(sqrt_discriminant, b)
93:         ta = -aux / (2*a)
94:         tb = -2*c / aux
95:         return sorted([ta, tb])
96: 
97:     def solve(self, trust_radius):
98:         raise NotImplementedError('The solve method should be implemented by '
99:                                   'the child class')
100: 
101: 
102: def _minimize_trust_region(fun, x0, args=(), jac=None, hess=None, hessp=None,
103:                            subproblem=None, initial_trust_radius=1.0,
104:                            max_trust_radius=1000.0, eta=0.15, gtol=1e-4,
105:                            maxiter=None, disp=False, return_all=False,
106:                            callback=None, inexact=True, **unknown_options):
107:     '''
108:     Minimization of scalar function of one or more variables using a
109:     trust-region algorithm.
110: 
111:     Options for the trust-region algorithm are:
112:         initial_trust_radius : float
113:             Initial trust radius.
114:         max_trust_radius : float
115:             Never propose steps that are longer than this value.
116:         eta : float
117:             Trust region related acceptance stringency for proposed steps.
118:         gtol : float
119:             Gradient norm must be less than `gtol`
120:             before successful termination.
121:         maxiter : int
122:             Maximum number of iterations to perform.
123:         disp : bool
124:             If True, print convergence message.
125:         inexact : bool
126:             Accuracy to solve subproblems. If True requires less nonlinear
127:             iterations, but more vector products. Only effective for method
128:             trust-krylov.
129: 
130:     This function is called by the `minimize` function.
131:     It is not supposed to be called directly.
132:     '''
133:     _check_unknown_options(unknown_options)
134: 
135:     if jac is None:
136:         raise ValueError('Jacobian is currently required for trust-region '
137:                          'methods')
138:     if hess is None and hessp is None:
139:         raise ValueError('Either the Hessian or the Hessian-vector product '
140:                          'is currently required for trust-region methods')
141:     if subproblem is None:
142:         raise ValueError('A subproblem solving strategy is required for '
143:                          'trust-region methods')
144:     if not (0 <= eta < 0.25):
145:         raise Exception('invalid acceptance stringency')
146:     if max_trust_radius <= 0:
147:         raise Exception('the max trust radius must be positive')
148:     if initial_trust_radius <= 0:
149:         raise ValueError('the initial trust radius must be positive')
150:     if initial_trust_radius >= max_trust_radius:
151:         raise ValueError('the initial trust radius must be less than the '
152:                          'max trust radius')
153: 
154:     # force the initial guess into a nice format
155:     x0 = np.asarray(x0).flatten()
156: 
157:     # Wrap the functions, for a couple reasons.
158:     # This tracks how many times they have been called
159:     # and it automatically passes the args.
160:     nfun, fun = wrap_function(fun, args)
161:     njac, jac = wrap_function(jac, args)
162:     nhess, hess = wrap_function(hess, args)
163:     nhessp, hessp = wrap_function(hessp, args)
164: 
165:     # limit the number of iterations
166:     if maxiter is None:
167:         maxiter = len(x0)*200
168: 
169:     # init the search status
170:     warnflag = 0
171: 
172:     # initialize the search
173:     trust_radius = initial_trust_radius
174:     x = x0
175:     if return_all:
176:         allvecs = [x]
177:     m = subproblem(x, fun, jac, hess, hessp)
178:     k = 0
179: 
180:     # search for the function min
181:     while True:
182: 
183:         # Solve the sub-problem.
184:         # This gives us the proposed step relative to the current position
185:         # and it tells us whether the proposed step
186:         # has reached the trust region boundary or not.
187:         try:
188:             p, hits_boundary = m.solve(trust_radius)
189:         except np.linalg.linalg.LinAlgError as e:
190:             warnflag = 3
191:             break
192: 
193:         # calculate the predicted value at the proposed point
194:         predicted_value = m(p)
195: 
196:         # define the local approximation at the proposed point
197:         x_proposed = x + p
198:         m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)
199: 
200:         # evaluate the ratio defined in equation (4.4)
201:         actual_reduction = m.fun - m_proposed.fun
202:         predicted_reduction = m.fun - predicted_value
203:         if predicted_reduction <= 0:
204:             warnflag = 2
205:             break
206:         rho = actual_reduction / predicted_reduction
207: 
208:         # update the trust radius according to the actual/predicted ratio
209:         if rho < 0.25:
210:             trust_radius *= 0.25
211:         elif rho > 0.75 and hits_boundary:
212:             trust_radius = min(2*trust_radius, max_trust_radius)
213: 
214:         # if the ratio is high enough then accept the proposed step
215:         if rho > eta:
216:             x = x_proposed
217:             m = m_proposed
218: 
219:         # append the best guess, call back, increment the iteration count
220:         if return_all:
221:             allvecs.append(x)
222:         if callback is not None:
223:             callback(x)
224:         k += 1
225: 
226:         # check if the gradient is small enough to stop
227:         if m.jac_mag < gtol:
228:             warnflag = 0
229:             break
230: 
231:         # check if we have looked at enough iterations
232:         if k >= maxiter:
233:             warnflag = 1
234:             break
235: 
236:     # print some stuff if requested
237:     status_messages = (
238:             _status_message['success'],
239:             _status_message['maxiter'],
240:             'A bad approximation caused failure to predict improvement.',
241:             'A linalg error occurred, such as a non-psd Hessian.',
242:             )
243:     if disp:
244:         if warnflag == 0:
245:             print(status_messages[warnflag])
246:         else:
247:             print('Warning: ' + status_messages[warnflag])
248:         print("         Current function value: %f" % m.fun)
249:         print("         Iterations: %d" % k)
250:         print("         Function evaluations: %d" % nfun[0])
251:         print("         Gradient evaluations: %d" % njac[0])
252:         print("         Hessian evaluations: %d" % nhess[0])
253: 
254:     result = OptimizeResult(x=x, success=(warnflag == 0), status=warnflag,
255:                             fun=m.fun, jac=m.jac, nfev=nfun[0], njev=njac[0],
256:                             nhev=nhess[0], nit=k,
257:                             message=status_messages[warnflag])
258: 
259:     if hess is not None:
260:         result['hess'] = m.hess
261: 
262:     if return_all:
263:         result['allvecs'] = allvecs
264: 
265:     return result
266: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_202455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Trust-region optimization.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import math' statement (line 4)
import math

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_202456 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_202456) is not StypyTypeError):

    if (import_202456 != 'pyd_module'):
        __import__(import_202456)
        sys_modules_202457 = sys.modules[import_202456]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_202457.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_202456)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import scipy.linalg' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_202458 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg')

if (type(import_202458) is not StypyTypeError):

    if (import_202458 != 'pyd_module'):
        __import__(import_202458)
        sys_modules_202459 = sys.modules[import_202458]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', sys_modules_202459.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', import_202458)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize.optimize import _check_unknown_options, wrap_function, _status_message, OptimizeResult' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_202460 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize.optimize')

if (type(import_202460) is not StypyTypeError):

    if (import_202460 != 'pyd_module'):
        __import__(import_202460)
        sys_modules_202461 = sys.modules[import_202460]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize.optimize', sys_modules_202461.module_type_store, module_type_store, ['_check_unknown_options', 'wrap_function', '_status_message', 'OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_202461, sys_modules_202461.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import _check_unknown_options, wrap_function, _status_message, OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize.optimize', None, module_type_store, ['_check_unknown_options', 'wrap_function', '_status_message', 'OptimizeResult'], [_check_unknown_options, wrap_function, _status_message, OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize.optimize', import_202460)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 11):

# Assigning a List to a Name (line 11):
__all__ = []
module_type_store.set_exportable_members([])

# Obtaining an instance of the builtin type 'list' (line 11)
list_202462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)

# Assigning a type to the variable '__all__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__all__', list_202462)
# Declaration of the 'BaseQuadraticSubproblem' class

class BaseQuadraticSubproblem(object, ):
    str_202463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n    Base/abstract class defining the quadratic model for trust-region\n    minimization. Child classes must implement the ``solve`` method.\n\n    Values of the objective function, jacobian and hessian (if provided) at\n    the current iterate ``x`` are evaluated on demand and then stored as\n    attributes ``fun``, ``jac``, ``hess``.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 24)
        None_202464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'None')
        # Getting the type of 'None' (line 24)
        None_202465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 53), 'None')
        defaults = [None_202464, None_202465]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.__init__', ['x', 'fun', 'jac', 'hess', 'hessp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'fun', 'jac', 'hess', 'hessp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 25):
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'x' (line 25)
        x_202466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'x')
        # Getting the type of 'self' (line 25)
        self_202467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member '_x' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_202467, '_x', x_202466)
        
        # Assigning a Name to a Attribute (line 26):
        
        # Assigning a Name to a Attribute (line 26):
        # Getting the type of 'None' (line 26)
        None_202468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'None')
        # Getting the type of 'self' (line 26)
        self_202469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member '_f' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_202469, '_f', None_202468)
        
        # Assigning a Name to a Attribute (line 27):
        
        # Assigning a Name to a Attribute (line 27):
        # Getting the type of 'None' (line 27)
        None_202470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'None')
        # Getting the type of 'self' (line 27)
        self_202471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member '_g' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_202471, '_g', None_202470)
        
        # Assigning a Name to a Attribute (line 28):
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'None' (line 28)
        None_202472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'None')
        # Getting the type of 'self' (line 28)
        self_202473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member '_h' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_202473, '_h', None_202472)
        
        # Assigning a Name to a Attribute (line 29):
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'None' (line 29)
        None_202474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'None')
        # Getting the type of 'self' (line 29)
        self_202475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member '_g_mag' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_202475, '_g_mag', None_202474)
        
        # Assigning a Name to a Attribute (line 30):
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'None' (line 30)
        None_202476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'None')
        # Getting the type of 'self' (line 30)
        self_202477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member '_cauchy_point' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_202477, '_cauchy_point', None_202476)
        
        # Assigning a Name to a Attribute (line 31):
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'None' (line 31)
        None_202478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'None')
        # Getting the type of 'self' (line 31)
        self_202479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member '_newton_point' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_202479, '_newton_point', None_202478)
        
        # Assigning a Name to a Attribute (line 32):
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'fun' (line 32)
        fun_202480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'fun')
        # Getting the type of 'self' (line 32)
        self_202481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member '_fun' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_202481, '_fun', fun_202480)
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'jac' (line 33)
        jac_202482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'jac')
        # Getting the type of 'self' (line 33)
        self_202483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member '_jac' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_202483, '_jac', jac_202482)
        
        # Assigning a Name to a Attribute (line 34):
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'hess' (line 34)
        hess_202484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'hess')
        # Getting the type of 'self' (line 34)
        self_202485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member '_hess' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_202485, '_hess', hess_202484)
        
        # Assigning a Name to a Attribute (line 35):
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'hessp' (line 35)
        hessp_202486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'hessp')
        # Getting the type of 'self' (line 35)
        self_202487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member '_hessp' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_202487, '_hessp', hessp_202486)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_localization', localization)
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_function_name', 'BaseQuadraticSubproblem.__call__')
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_param_names_list', ['p'])
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseQuadraticSubproblem.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.__call__', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        # Getting the type of 'self' (line 38)
        self_202488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'self')
        # Obtaining the member 'fun' of a type (line 38)
        fun_202489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), self_202488, 'fun')
        
        # Call to dot(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_202492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'self', False)
        # Obtaining the member 'jac' of a type (line 38)
        jac_202493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 33), self_202492, 'jac')
        # Getting the type of 'p' (line 38)
        p_202494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 43), 'p', False)
        # Processing the call keyword arguments (line 38)
        kwargs_202495 = {}
        # Getting the type of 'np' (line 38)
        np_202490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'np', False)
        # Obtaining the member 'dot' of a type (line 38)
        dot_202491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 26), np_202490, 'dot')
        # Calling dot(args, kwargs) (line 38)
        dot_call_result_202496 = invoke(stypy.reporting.localization.Localization(__file__, 38, 26), dot_202491, *[jac_202493, p_202494], **kwargs_202495)
        
        # Applying the binary operator '+' (line 38)
        result_add_202497 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 15), '+', fun_202489, dot_call_result_202496)
        
        float_202498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 48), 'float')
        
        # Call to dot(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'p' (line 38)
        p_202501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 61), 'p', False)
        
        # Call to hessp(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'p' (line 38)
        p_202504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 75), 'p', False)
        # Processing the call keyword arguments (line 38)
        kwargs_202505 = {}
        # Getting the type of 'self' (line 38)
        self_202502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 64), 'self', False)
        # Obtaining the member 'hessp' of a type (line 38)
        hessp_202503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 64), self_202502, 'hessp')
        # Calling hessp(args, kwargs) (line 38)
        hessp_call_result_202506 = invoke(stypy.reporting.localization.Localization(__file__, 38, 64), hessp_202503, *[p_202504], **kwargs_202505)
        
        # Processing the call keyword arguments (line 38)
        kwargs_202507 = {}
        # Getting the type of 'np' (line 38)
        np_202499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 54), 'np', False)
        # Obtaining the member 'dot' of a type (line 38)
        dot_202500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 54), np_202499, 'dot')
        # Calling dot(args, kwargs) (line 38)
        dot_call_result_202508 = invoke(stypy.reporting.localization.Localization(__file__, 38, 54), dot_202500, *[p_202501, hessp_call_result_202506], **kwargs_202507)
        
        # Applying the binary operator '*' (line 38)
        result_mul_202509 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 48), '*', float_202498, dot_call_result_202508)
        
        # Applying the binary operator '+' (line 38)
        result_add_202510 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 46), '+', result_add_202497, result_mul_202509)
        
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', result_add_202510)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_202511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_202511


    @norecursion
    def fun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun'
        module_type_store = module_type_store.open_function_context('fun', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_localization', localization)
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_function_name', 'BaseQuadraticSubproblem.fun')
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_param_names_list', [])
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseQuadraticSubproblem.fun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.fun', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun(...)' code ##################

        str_202512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'str', 'Value of objective function at current iteration.')
        
        # Type idiom detected: calculating its left and rigth part (line 43)
        # Getting the type of 'self' (line 43)
        self_202513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'self')
        # Obtaining the member '_f' of a type (line 43)
        _f_202514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 11), self_202513, '_f')
        # Getting the type of 'None' (line 43)
        None_202515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'None')
        
        (may_be_202516, more_types_in_union_202517) = may_be_none(_f_202514, None_202515)

        if may_be_202516:

            if more_types_in_union_202517:
                # Runtime conditional SSA (line 43)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 44):
            
            # Assigning a Call to a Attribute (line 44):
            
            # Call to _fun(...): (line 44)
            # Processing the call arguments (line 44)
            # Getting the type of 'self' (line 44)
            self_202520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 32), 'self', False)
            # Obtaining the member '_x' of a type (line 44)
            _x_202521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 32), self_202520, '_x')
            # Processing the call keyword arguments (line 44)
            kwargs_202522 = {}
            # Getting the type of 'self' (line 44)
            self_202518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'self', False)
            # Obtaining the member '_fun' of a type (line 44)
            _fun_202519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), self_202518, '_fun')
            # Calling _fun(args, kwargs) (line 44)
            _fun_call_result_202523 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), _fun_202519, *[_x_202521], **kwargs_202522)
            
            # Getting the type of 'self' (line 44)
            self_202524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'self')
            # Setting the type of the member '_f' of a type (line 44)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), self_202524, '_f', _fun_call_result_202523)

            if more_types_in_union_202517:
                # SSA join for if statement (line 43)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 45)
        self_202525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'self')
        # Obtaining the member '_f' of a type (line 45)
        _f_202526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), self_202525, '_f')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', _f_202526)
        
        # ################# End of 'fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_202527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202527)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun'
        return stypy_return_type_202527


    @norecursion
    def jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac'
        module_type_store = module_type_store.open_function_context('jac', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_localization', localization)
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_function_name', 'BaseQuadraticSubproblem.jac')
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_param_names_list', [])
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseQuadraticSubproblem.jac.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.jac', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac(...)' code ##################

        str_202528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'str', 'Value of jacobian of objective function at current iteration.')
        
        # Type idiom detected: calculating its left and rigth part (line 50)
        # Getting the type of 'self' (line 50)
        self_202529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'self')
        # Obtaining the member '_g' of a type (line 50)
        _g_202530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 11), self_202529, '_g')
        # Getting the type of 'None' (line 50)
        None_202531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'None')
        
        (may_be_202532, more_types_in_union_202533) = may_be_none(_g_202530, None_202531)

        if may_be_202532:

            if more_types_in_union_202533:
                # Runtime conditional SSA (line 50)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 51):
            
            # Assigning a Call to a Attribute (line 51):
            
            # Call to _jac(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'self' (line 51)
            self_202536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'self', False)
            # Obtaining the member '_x' of a type (line 51)
            _x_202537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 32), self_202536, '_x')
            # Processing the call keyword arguments (line 51)
            kwargs_202538 = {}
            # Getting the type of 'self' (line 51)
            self_202534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'self', False)
            # Obtaining the member '_jac' of a type (line 51)
            _jac_202535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 22), self_202534, '_jac')
            # Calling _jac(args, kwargs) (line 51)
            _jac_call_result_202539 = invoke(stypy.reporting.localization.Localization(__file__, 51, 22), _jac_202535, *[_x_202537], **kwargs_202538)
            
            # Getting the type of 'self' (line 51)
            self_202540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
            # Setting the type of the member '_g' of a type (line 51)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_202540, '_g', _jac_call_result_202539)

            if more_types_in_union_202533:
                # SSA join for if statement (line 50)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 52)
        self_202541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'self')
        # Obtaining the member '_g' of a type (line 52)
        _g_202542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), self_202541, '_g')
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', _g_202542)
        
        # ################# End of 'jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_202543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202543)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac'
        return stypy_return_type_202543


    @norecursion
    def hess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hess'
        module_type_store = module_type_store.open_function_context('hess', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_localization', localization)
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_function_name', 'BaseQuadraticSubproblem.hess')
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_param_names_list', [])
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseQuadraticSubproblem.hess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.hess', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hess(...)' code ##################

        str_202544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'str', 'Value of hessian of objective function at current iteration.')
        
        # Type idiom detected: calculating its left and rigth part (line 57)
        # Getting the type of 'self' (line 57)
        self_202545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'self')
        # Obtaining the member '_h' of a type (line 57)
        _h_202546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), self_202545, '_h')
        # Getting the type of 'None' (line 57)
        None_202547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 22), 'None')
        
        (may_be_202548, more_types_in_union_202549) = may_be_none(_h_202546, None_202547)

        if may_be_202548:

            if more_types_in_union_202549:
                # Runtime conditional SSA (line 57)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 58):
            
            # Assigning a Call to a Attribute (line 58):
            
            # Call to _hess(...): (line 58)
            # Processing the call arguments (line 58)
            # Getting the type of 'self' (line 58)
            self_202552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 33), 'self', False)
            # Obtaining the member '_x' of a type (line 58)
            _x_202553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 33), self_202552, '_x')
            # Processing the call keyword arguments (line 58)
            kwargs_202554 = {}
            # Getting the type of 'self' (line 58)
            self_202550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'self', False)
            # Obtaining the member '_hess' of a type (line 58)
            _hess_202551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 22), self_202550, '_hess')
            # Calling _hess(args, kwargs) (line 58)
            _hess_call_result_202555 = invoke(stypy.reporting.localization.Localization(__file__, 58, 22), _hess_202551, *[_x_202553], **kwargs_202554)
            
            # Getting the type of 'self' (line 58)
            self_202556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'self')
            # Setting the type of the member '_h' of a type (line 58)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), self_202556, '_h', _hess_call_result_202555)

            if more_types_in_union_202549:
                # SSA join for if statement (line 57)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 59)
        self_202557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'self')
        # Obtaining the member '_h' of a type (line 59)
        _h_202558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), self_202557, '_h')
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', _h_202558)
        
        # ################# End of 'hess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hess' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_202559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202559)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hess'
        return stypy_return_type_202559


    @norecursion
    def hessp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hessp'
        module_type_store = module_type_store.open_function_context('hessp', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_localization', localization)
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_function_name', 'BaseQuadraticSubproblem.hessp')
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_param_names_list', ['p'])
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseQuadraticSubproblem.hessp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.hessp', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hessp', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hessp(...)' code ##################

        
        
        # Getting the type of 'self' (line 62)
        self_202560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'self')
        # Obtaining the member '_hessp' of a type (line 62)
        _hessp_202561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 11), self_202560, '_hessp')
        # Getting the type of 'None' (line 62)
        None_202562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'None')
        # Applying the binary operator 'isnot' (line 62)
        result_is_not_202563 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), 'isnot', _hessp_202561, None_202562)
        
        # Testing the type of an if condition (line 62)
        if_condition_202564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_is_not_202563)
        # Assigning a type to the variable 'if_condition_202564' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_202564', if_condition_202564)
        # SSA begins for if statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _hessp(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_202567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'self', False)
        # Obtaining the member '_x' of a type (line 63)
        _x_202568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 31), self_202567, '_x')
        # Getting the type of 'p' (line 63)
        p_202569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'p', False)
        # Processing the call keyword arguments (line 63)
        kwargs_202570 = {}
        # Getting the type of 'self' (line 63)
        self_202565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'self', False)
        # Obtaining the member '_hessp' of a type (line 63)
        _hessp_202566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), self_202565, '_hessp')
        # Calling _hessp(args, kwargs) (line 63)
        _hessp_call_result_202571 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), _hessp_202566, *[_x_202568, p_202569], **kwargs_202570)
        
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', _hessp_call_result_202571)
        # SSA branch for the else part of an if statement (line 62)
        module_type_store.open_ssa_branch('else')
        
        # Call to dot(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_202574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'self', False)
        # Obtaining the member 'hess' of a type (line 65)
        hess_202575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), self_202574, 'hess')
        # Getting the type of 'p' (line 65)
        p_202576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 37), 'p', False)
        # Processing the call keyword arguments (line 65)
        kwargs_202577 = {}
        # Getting the type of 'np' (line 65)
        np_202572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'np', False)
        # Obtaining the member 'dot' of a type (line 65)
        dot_202573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), np_202572, 'dot')
        # Calling dot(args, kwargs) (line 65)
        dot_call_result_202578 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), dot_202573, *[hess_202575, p_202576], **kwargs_202577)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', dot_call_result_202578)
        # SSA join for if statement (line 62)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'hessp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hessp' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_202579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hessp'
        return stypy_return_type_202579


    @norecursion
    def jac_mag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_mag'
        module_type_store = module_type_store.open_function_context('jac_mag', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_localization', localization)
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_function_name', 'BaseQuadraticSubproblem.jac_mag')
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_param_names_list', [])
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseQuadraticSubproblem.jac_mag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.jac_mag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_mag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_mag(...)' code ##################

        str_202580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'str', 'Magniture of jacobian of objective function at current iteration.')
        
        # Type idiom detected: calculating its left and rigth part (line 70)
        # Getting the type of 'self' (line 70)
        self_202581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'self')
        # Obtaining the member '_g_mag' of a type (line 70)
        _g_mag_202582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 11), self_202581, '_g_mag')
        # Getting the type of 'None' (line 70)
        None_202583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'None')
        
        (may_be_202584, more_types_in_union_202585) = may_be_none(_g_mag_202582, None_202583)

        if may_be_202584:

            if more_types_in_union_202585:
                # Runtime conditional SSA (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 71):
            
            # Assigning a Call to a Attribute (line 71):
            
            # Call to norm(...): (line 71)
            # Processing the call arguments (line 71)
            # Getting the type of 'self' (line 71)
            self_202589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 44), 'self', False)
            # Obtaining the member 'jac' of a type (line 71)
            jac_202590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 44), self_202589, 'jac')
            # Processing the call keyword arguments (line 71)
            kwargs_202591 = {}
            # Getting the type of 'scipy' (line 71)
            scipy_202586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'scipy', False)
            # Obtaining the member 'linalg' of a type (line 71)
            linalg_202587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), scipy_202586, 'linalg')
            # Obtaining the member 'norm' of a type (line 71)
            norm_202588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), linalg_202587, 'norm')
            # Calling norm(args, kwargs) (line 71)
            norm_call_result_202592 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), norm_202588, *[jac_202590], **kwargs_202591)
            
            # Getting the type of 'self' (line 71)
            self_202593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'self')
            # Setting the type of the member '_g_mag' of a type (line 71)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), self_202593, '_g_mag', norm_call_result_202592)

            if more_types_in_union_202585:
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 72)
        self_202594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'self')
        # Obtaining the member '_g_mag' of a type (line 72)
        _g_mag_202595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), self_202594, '_g_mag')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', _g_mag_202595)
        
        # ################# End of 'jac_mag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_mag' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_202596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202596)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_mag'
        return stypy_return_type_202596


    @norecursion
    def get_boundaries_intersections(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_boundaries_intersections'
        module_type_store = module_type_store.open_function_context('get_boundaries_intersections', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_localization', localization)
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_function_name', 'BaseQuadraticSubproblem.get_boundaries_intersections')
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_param_names_list', ['z', 'd', 'trust_radius'])
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseQuadraticSubproblem.get_boundaries_intersections.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.get_boundaries_intersections', ['z', 'd', 'trust_radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_boundaries_intersections', localization, ['z', 'd', 'trust_radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_boundaries_intersections(...)' code ##################

        str_202597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', '\n        Solve the scalar quadratic equation ||z + t d|| == trust_radius.\n        This is like a line-sphere intersection.\n        Return the two values of t, sorted from low to high.\n        ')
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to dot(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'd' (line 80)
        d_202600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'd', False)
        # Getting the type of 'd' (line 80)
        d_202601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'd', False)
        # Processing the call keyword arguments (line 80)
        kwargs_202602 = {}
        # Getting the type of 'np' (line 80)
        np_202598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 80)
        dot_202599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), np_202598, 'dot')
        # Calling dot(args, kwargs) (line 80)
        dot_call_result_202603 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), dot_202599, *[d_202600, d_202601], **kwargs_202602)
        
        # Assigning a type to the variable 'a' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'a', dot_call_result_202603)
        
        # Assigning a BinOp to a Name (line 81):
        
        # Assigning a BinOp to a Name (line 81):
        int_202604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 12), 'int')
        
        # Call to dot(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'z' (line 81)
        z_202607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'z', False)
        # Getting the type of 'd' (line 81)
        d_202608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'd', False)
        # Processing the call keyword arguments (line 81)
        kwargs_202609 = {}
        # Getting the type of 'np' (line 81)
        np_202605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'np', False)
        # Obtaining the member 'dot' of a type (line 81)
        dot_202606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 16), np_202605, 'dot')
        # Calling dot(args, kwargs) (line 81)
        dot_call_result_202610 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), dot_202606, *[z_202607, d_202608], **kwargs_202609)
        
        # Applying the binary operator '*' (line 81)
        result_mul_202611 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 12), '*', int_202604, dot_call_result_202610)
        
        # Assigning a type to the variable 'b' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'b', result_mul_202611)
        
        # Assigning a BinOp to a Name (line 82):
        
        # Assigning a BinOp to a Name (line 82):
        
        # Call to dot(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'z' (line 82)
        z_202614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'z', False)
        # Getting the type of 'z' (line 82)
        z_202615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'z', False)
        # Processing the call keyword arguments (line 82)
        kwargs_202616 = {}
        # Getting the type of 'np' (line 82)
        np_202612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 82)
        dot_202613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), np_202612, 'dot')
        # Calling dot(args, kwargs) (line 82)
        dot_call_result_202617 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), dot_202613, *[z_202614, z_202615], **kwargs_202616)
        
        # Getting the type of 'trust_radius' (line 82)
        trust_radius_202618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'trust_radius')
        int_202619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 41), 'int')
        # Applying the binary operator '**' (line 82)
        result_pow_202620 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 27), '**', trust_radius_202618, int_202619)
        
        # Applying the binary operator '-' (line 82)
        result_sub_202621 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 12), '-', dot_call_result_202617, result_pow_202620)
        
        # Assigning a type to the variable 'c' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'c', result_sub_202621)
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to sqrt(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'b' (line 83)
        b_202624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 38), 'b', False)
        # Getting the type of 'b' (line 83)
        b_202625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'b', False)
        # Applying the binary operator '*' (line 83)
        result_mul_202626 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 38), '*', b_202624, b_202625)
        
        int_202627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 44), 'int')
        # Getting the type of 'a' (line 83)
        a_202628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'a', False)
        # Applying the binary operator '*' (line 83)
        result_mul_202629 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 44), '*', int_202627, a_202628)
        
        # Getting the type of 'c' (line 83)
        c_202630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 48), 'c', False)
        # Applying the binary operator '*' (line 83)
        result_mul_202631 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 47), '*', result_mul_202629, c_202630)
        
        # Applying the binary operator '-' (line 83)
        result_sub_202632 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 38), '-', result_mul_202626, result_mul_202631)
        
        # Processing the call keyword arguments (line 83)
        kwargs_202633 = {}
        # Getting the type of 'math' (line 83)
        math_202622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 83)
        sqrt_202623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 28), math_202622, 'sqrt')
        # Calling sqrt(args, kwargs) (line 83)
        sqrt_call_result_202634 = invoke(stypy.reporting.localization.Localization(__file__, 83, 28), sqrt_202623, *[result_sub_202632], **kwargs_202633)
        
        # Assigning a type to the variable 'sqrt_discriminant' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'sqrt_discriminant', sqrt_call_result_202634)
        
        # Assigning a BinOp to a Name (line 92):
        
        # Assigning a BinOp to a Name (line 92):
        # Getting the type of 'b' (line 92)
        b_202635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'b')
        
        # Call to copysign(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'sqrt_discriminant' (line 92)
        sqrt_discriminant_202638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'sqrt_discriminant', False)
        # Getting the type of 'b' (line 92)
        b_202639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 51), 'b', False)
        # Processing the call keyword arguments (line 92)
        kwargs_202640 = {}
        # Getting the type of 'math' (line 92)
        math_202636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'math', False)
        # Obtaining the member 'copysign' of a type (line 92)
        copysign_202637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 18), math_202636, 'copysign')
        # Calling copysign(args, kwargs) (line 92)
        copysign_call_result_202641 = invoke(stypy.reporting.localization.Localization(__file__, 92, 18), copysign_202637, *[sqrt_discriminant_202638, b_202639], **kwargs_202640)
        
        # Applying the binary operator '+' (line 92)
        result_add_202642 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 14), '+', b_202635, copysign_call_result_202641)
        
        # Assigning a type to the variable 'aux' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'aux', result_add_202642)
        
        # Assigning a BinOp to a Name (line 93):
        
        # Assigning a BinOp to a Name (line 93):
        
        # Getting the type of 'aux' (line 93)
        aux_202643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'aux')
        # Applying the 'usub' unary operator (line 93)
        result___neg___202644 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 13), 'usub', aux_202643)
        
        int_202645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'int')
        # Getting the type of 'a' (line 93)
        a_202646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'a')
        # Applying the binary operator '*' (line 93)
        result_mul_202647 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 21), '*', int_202645, a_202646)
        
        # Applying the binary operator 'div' (line 93)
        result_div_202648 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 13), 'div', result___neg___202644, result_mul_202647)
        
        # Assigning a type to the variable 'ta' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'ta', result_div_202648)
        
        # Assigning a BinOp to a Name (line 94):
        
        # Assigning a BinOp to a Name (line 94):
        int_202649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 13), 'int')
        # Getting the type of 'c' (line 94)
        c_202650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'c')
        # Applying the binary operator '*' (line 94)
        result_mul_202651 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 13), '*', int_202649, c_202650)
        
        # Getting the type of 'aux' (line 94)
        aux_202652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'aux')
        # Applying the binary operator 'div' (line 94)
        result_div_202653 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 18), 'div', result_mul_202651, aux_202652)
        
        # Assigning a type to the variable 'tb' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tb', result_div_202653)
        
        # Call to sorted(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_202655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        # Getting the type of 'ta' (line 95)
        ta_202656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'ta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 22), list_202655, ta_202656)
        # Adding element type (line 95)
        # Getting the type of 'tb' (line 95)
        tb_202657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 27), 'tb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 22), list_202655, tb_202657)
        
        # Processing the call keyword arguments (line 95)
        kwargs_202658 = {}
        # Getting the type of 'sorted' (line 95)
        sorted_202654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'sorted', False)
        # Calling sorted(args, kwargs) (line 95)
        sorted_call_result_202659 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), sorted_202654, *[list_202655], **kwargs_202658)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', sorted_call_result_202659)
        
        # ################# End of 'get_boundaries_intersections(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_boundaries_intersections' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_202660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_boundaries_intersections'
        return stypy_return_type_202660


    @norecursion
    def solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_localization', localization)
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_function_name', 'BaseQuadraticSubproblem.solve')
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_param_names_list', ['trust_radius'])
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseQuadraticSubproblem.solve.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseQuadraticSubproblem.solve', ['trust_radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solve', localization, ['trust_radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solve(...)' code ##################

        
        # Call to NotImplementedError(...): (line 98)
        # Processing the call arguments (line 98)
        str_202662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 34), 'str', 'The solve method should be implemented by the child class')
        # Processing the call keyword arguments (line 98)
        kwargs_202663 = {}
        # Getting the type of 'NotImplementedError' (line 98)
        NotImplementedError_202661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 98)
        NotImplementedError_call_result_202664 = invoke(stypy.reporting.localization.Localization(__file__, 98, 14), NotImplementedError_202661, *[str_202662], **kwargs_202663)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 98, 8), NotImplementedError_call_result_202664, 'raise parameter', BaseException)
        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_202665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202665)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_202665


# Assigning a type to the variable 'BaseQuadraticSubproblem' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'BaseQuadraticSubproblem', BaseQuadraticSubproblem)

@norecursion
def _minimize_trust_region(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_202666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    
    # Getting the type of 'None' (line 102)
    None_202667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 49), 'None')
    # Getting the type of 'None' (line 102)
    None_202668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 60), 'None')
    # Getting the type of 'None' (line 102)
    None_202669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 72), 'None')
    # Getting the type of 'None' (line 103)
    None_202670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 'None')
    float_202671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 65), 'float')
    float_202672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 44), 'float')
    float_202673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 56), 'float')
    float_202674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 67), 'float')
    # Getting the type of 'None' (line 105)
    None_202675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'None')
    # Getting the type of 'False' (line 105)
    False_202676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 46), 'False')
    # Getting the type of 'False' (line 105)
    False_202677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 64), 'False')
    # Getting the type of 'None' (line 106)
    None_202678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'None')
    # Getting the type of 'True' (line 106)
    True_202679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 50), 'True')
    defaults = [tuple_202666, None_202667, None_202668, None_202669, None_202670, float_202671, float_202672, float_202673, float_202674, None_202675, False_202676, False_202677, None_202678, True_202679]
    # Create a new context for function '_minimize_trust_region'
    module_type_store = module_type_store.open_function_context('_minimize_trust_region', 102, 0, False)
    
    # Passed parameters checking function
    _minimize_trust_region.stypy_localization = localization
    _minimize_trust_region.stypy_type_of_self = None
    _minimize_trust_region.stypy_type_store = module_type_store
    _minimize_trust_region.stypy_function_name = '_minimize_trust_region'
    _minimize_trust_region.stypy_param_names_list = ['fun', 'x0', 'args', 'jac', 'hess', 'hessp', 'subproblem', 'initial_trust_radius', 'max_trust_radius', 'eta', 'gtol', 'maxiter', 'disp', 'return_all', 'callback', 'inexact']
    _minimize_trust_region.stypy_varargs_param_name = None
    _minimize_trust_region.stypy_kwargs_param_name = 'unknown_options'
    _minimize_trust_region.stypy_call_defaults = defaults
    _minimize_trust_region.stypy_call_varargs = varargs
    _minimize_trust_region.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minimize_trust_region', ['fun', 'x0', 'args', 'jac', 'hess', 'hessp', 'subproblem', 'initial_trust_radius', 'max_trust_radius', 'eta', 'gtol', 'maxiter', 'disp', 'return_all', 'callback', 'inexact'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minimize_trust_region', localization, ['fun', 'x0', 'args', 'jac', 'hess', 'hessp', 'subproblem', 'initial_trust_radius', 'max_trust_radius', 'eta', 'gtol', 'maxiter', 'disp', 'return_all', 'callback', 'inexact'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minimize_trust_region(...)' code ##################

    str_202680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', '\n    Minimization of scalar function of one or more variables using a\n    trust-region algorithm.\n\n    Options for the trust-region algorithm are:\n        initial_trust_radius : float\n            Initial trust radius.\n        max_trust_radius : float\n            Never propose steps that are longer than this value.\n        eta : float\n            Trust region related acceptance stringency for proposed steps.\n        gtol : float\n            Gradient norm must be less than `gtol`\n            before successful termination.\n        maxiter : int\n            Maximum number of iterations to perform.\n        disp : bool\n            If True, print convergence message.\n        inexact : bool\n            Accuracy to solve subproblems. If True requires less nonlinear\n            iterations, but more vector products. Only effective for method\n            trust-krylov.\n\n    This function is called by the `minimize` function.\n    It is not supposed to be called directly.\n    ')
    
    # Call to _check_unknown_options(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'unknown_options' (line 133)
    unknown_options_202682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 133)
    kwargs_202683 = {}
    # Getting the type of '_check_unknown_options' (line 133)
    _check_unknown_options_202681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 133)
    _check_unknown_options_call_result_202684 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), _check_unknown_options_202681, *[unknown_options_202682], **kwargs_202683)
    
    
    # Type idiom detected: calculating its left and rigth part (line 135)
    # Getting the type of 'jac' (line 135)
    jac_202685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 7), 'jac')
    # Getting the type of 'None' (line 135)
    None_202686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'None')
    
    (may_be_202687, more_types_in_union_202688) = may_be_none(jac_202685, None_202686)

    if may_be_202687:

        if more_types_in_union_202688:
            # Runtime conditional SSA (line 135)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 136)
        # Processing the call arguments (line 136)
        str_202690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 25), 'str', 'Jacobian is currently required for trust-region methods')
        # Processing the call keyword arguments (line 136)
        kwargs_202691 = {}
        # Getting the type of 'ValueError' (line 136)
        ValueError_202689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 136)
        ValueError_call_result_202692 = invoke(stypy.reporting.localization.Localization(__file__, 136, 14), ValueError_202689, *[str_202690], **kwargs_202691)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 136, 8), ValueError_call_result_202692, 'raise parameter', BaseException)

        if more_types_in_union_202688:
            # SSA join for if statement (line 135)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'hess' (line 138)
    hess_202693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 7), 'hess')
    # Getting the type of 'None' (line 138)
    None_202694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'None')
    # Applying the binary operator 'is' (line 138)
    result_is__202695 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 7), 'is', hess_202693, None_202694)
    
    
    # Getting the type of 'hessp' (line 138)
    hessp_202696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'hessp')
    # Getting the type of 'None' (line 138)
    None_202697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 33), 'None')
    # Applying the binary operator 'is' (line 138)
    result_is__202698 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 24), 'is', hessp_202696, None_202697)
    
    # Applying the binary operator 'and' (line 138)
    result_and_keyword_202699 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 7), 'and', result_is__202695, result_is__202698)
    
    # Testing the type of an if condition (line 138)
    if_condition_202700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 4), result_and_keyword_202699)
    # Assigning a type to the variable 'if_condition_202700' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'if_condition_202700', if_condition_202700)
    # SSA begins for if statement (line 138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 139)
    # Processing the call arguments (line 139)
    str_202702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 25), 'str', 'Either the Hessian or the Hessian-vector product is currently required for trust-region methods')
    # Processing the call keyword arguments (line 139)
    kwargs_202703 = {}
    # Getting the type of 'ValueError' (line 139)
    ValueError_202701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 139)
    ValueError_call_result_202704 = invoke(stypy.reporting.localization.Localization(__file__, 139, 14), ValueError_202701, *[str_202702], **kwargs_202703)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 8), ValueError_call_result_202704, 'raise parameter', BaseException)
    # SSA join for if statement (line 138)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 141)
    # Getting the type of 'subproblem' (line 141)
    subproblem_202705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 7), 'subproblem')
    # Getting the type of 'None' (line 141)
    None_202706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'None')
    
    (may_be_202707, more_types_in_union_202708) = may_be_none(subproblem_202705, None_202706)

    if may_be_202707:

        if more_types_in_union_202708:
            # Runtime conditional SSA (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 142)
        # Processing the call arguments (line 142)
        str_202710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 25), 'str', 'A subproblem solving strategy is required for trust-region methods')
        # Processing the call keyword arguments (line 142)
        kwargs_202711 = {}
        # Getting the type of 'ValueError' (line 142)
        ValueError_202709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 142)
        ValueError_call_result_202712 = invoke(stypy.reporting.localization.Localization(__file__, 142, 14), ValueError_202709, *[str_202710], **kwargs_202711)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 142, 8), ValueError_call_result_202712, 'raise parameter', BaseException)

        if more_types_in_union_202708:
            # SSA join for if statement (line 141)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    int_202713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 12), 'int')
    # Getting the type of 'eta' (line 144)
    eta_202714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 17), 'eta')
    # Applying the binary operator '<=' (line 144)
    result_le_202715 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), '<=', int_202713, eta_202714)
    float_202716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'float')
    # Applying the binary operator '<' (line 144)
    result_lt_202717 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), '<', eta_202714, float_202716)
    # Applying the binary operator '&' (line 144)
    result_and__202718 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), '&', result_le_202715, result_lt_202717)
    
    # Applying the 'not' unary operator (line 144)
    result_not__202719 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 7), 'not', result_and__202718)
    
    # Testing the type of an if condition (line 144)
    if_condition_202720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 4), result_not__202719)
    # Assigning a type to the variable 'if_condition_202720' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'if_condition_202720', if_condition_202720)
    # SSA begins for if statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 145)
    # Processing the call arguments (line 145)
    str_202722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 24), 'str', 'invalid acceptance stringency')
    # Processing the call keyword arguments (line 145)
    kwargs_202723 = {}
    # Getting the type of 'Exception' (line 145)
    Exception_202721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 145)
    Exception_call_result_202724 = invoke(stypy.reporting.localization.Localization(__file__, 145, 14), Exception_202721, *[str_202722], **kwargs_202723)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 145, 8), Exception_call_result_202724, 'raise parameter', BaseException)
    # SSA join for if statement (line 144)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'max_trust_radius' (line 146)
    max_trust_radius_202725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 7), 'max_trust_radius')
    int_202726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 27), 'int')
    # Applying the binary operator '<=' (line 146)
    result_le_202727 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 7), '<=', max_trust_radius_202725, int_202726)
    
    # Testing the type of an if condition (line 146)
    if_condition_202728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 4), result_le_202727)
    # Assigning a type to the variable 'if_condition_202728' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'if_condition_202728', if_condition_202728)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 147)
    # Processing the call arguments (line 147)
    str_202730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'str', 'the max trust radius must be positive')
    # Processing the call keyword arguments (line 147)
    kwargs_202731 = {}
    # Getting the type of 'Exception' (line 147)
    Exception_202729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 147)
    Exception_call_result_202732 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), Exception_202729, *[str_202730], **kwargs_202731)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 147, 8), Exception_call_result_202732, 'raise parameter', BaseException)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'initial_trust_radius' (line 148)
    initial_trust_radius_202733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'initial_trust_radius')
    int_202734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 31), 'int')
    # Applying the binary operator '<=' (line 148)
    result_le_202735 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 7), '<=', initial_trust_radius_202733, int_202734)
    
    # Testing the type of an if condition (line 148)
    if_condition_202736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), result_le_202735)
    # Assigning a type to the variable 'if_condition_202736' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_202736', if_condition_202736)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 149)
    # Processing the call arguments (line 149)
    str_202738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 25), 'str', 'the initial trust radius must be positive')
    # Processing the call keyword arguments (line 149)
    kwargs_202739 = {}
    # Getting the type of 'ValueError' (line 149)
    ValueError_202737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 149)
    ValueError_call_result_202740 = invoke(stypy.reporting.localization.Localization(__file__, 149, 14), ValueError_202737, *[str_202738], **kwargs_202739)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 149, 8), ValueError_call_result_202740, 'raise parameter', BaseException)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'initial_trust_radius' (line 150)
    initial_trust_radius_202741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'initial_trust_radius')
    # Getting the type of 'max_trust_radius' (line 150)
    max_trust_radius_202742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 31), 'max_trust_radius')
    # Applying the binary operator '>=' (line 150)
    result_ge_202743 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 7), '>=', initial_trust_radius_202741, max_trust_radius_202742)
    
    # Testing the type of an if condition (line 150)
    if_condition_202744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 4), result_ge_202743)
    # Assigning a type to the variable 'if_condition_202744' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'if_condition_202744', if_condition_202744)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 151)
    # Processing the call arguments (line 151)
    str_202746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 25), 'str', 'the initial trust radius must be less than the max trust radius')
    # Processing the call keyword arguments (line 151)
    kwargs_202747 = {}
    # Getting the type of 'ValueError' (line 151)
    ValueError_202745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 151)
    ValueError_call_result_202748 = invoke(stypy.reporting.localization.Localization(__file__, 151, 14), ValueError_202745, *[str_202746], **kwargs_202747)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 151, 8), ValueError_call_result_202748, 'raise parameter', BaseException)
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 155):
    
    # Assigning a Call to a Name (line 155):
    
    # Call to flatten(...): (line 155)
    # Processing the call keyword arguments (line 155)
    kwargs_202755 = {}
    
    # Call to asarray(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'x0' (line 155)
    x0_202751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'x0', False)
    # Processing the call keyword arguments (line 155)
    kwargs_202752 = {}
    # Getting the type of 'np' (line 155)
    np_202749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 9), 'np', False)
    # Obtaining the member 'asarray' of a type (line 155)
    asarray_202750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 9), np_202749, 'asarray')
    # Calling asarray(args, kwargs) (line 155)
    asarray_call_result_202753 = invoke(stypy.reporting.localization.Localization(__file__, 155, 9), asarray_202750, *[x0_202751], **kwargs_202752)
    
    # Obtaining the member 'flatten' of a type (line 155)
    flatten_202754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 9), asarray_call_result_202753, 'flatten')
    # Calling flatten(args, kwargs) (line 155)
    flatten_call_result_202756 = invoke(stypy.reporting.localization.Localization(__file__, 155, 9), flatten_202754, *[], **kwargs_202755)
    
    # Assigning a type to the variable 'x0' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'x0', flatten_call_result_202756)
    
    # Assigning a Call to a Tuple (line 160):
    
    # Assigning a Subscript to a Name (line 160):
    
    # Obtaining the type of the subscript
    int_202757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 4), 'int')
    
    # Call to wrap_function(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'fun' (line 160)
    fun_202759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 30), 'fun', False)
    # Getting the type of 'args' (line 160)
    args_202760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 35), 'args', False)
    # Processing the call keyword arguments (line 160)
    kwargs_202761 = {}
    # Getting the type of 'wrap_function' (line 160)
    wrap_function_202758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 160)
    wrap_function_call_result_202762 = invoke(stypy.reporting.localization.Localization(__file__, 160, 16), wrap_function_202758, *[fun_202759, args_202760], **kwargs_202761)
    
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___202763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 4), wrap_function_call_result_202762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_202764 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), getitem___202763, int_202757)
    
    # Assigning a type to the variable 'tuple_var_assignment_202445' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'tuple_var_assignment_202445', subscript_call_result_202764)
    
    # Assigning a Subscript to a Name (line 160):
    
    # Obtaining the type of the subscript
    int_202765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 4), 'int')
    
    # Call to wrap_function(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'fun' (line 160)
    fun_202767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 30), 'fun', False)
    # Getting the type of 'args' (line 160)
    args_202768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 35), 'args', False)
    # Processing the call keyword arguments (line 160)
    kwargs_202769 = {}
    # Getting the type of 'wrap_function' (line 160)
    wrap_function_202766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 160)
    wrap_function_call_result_202770 = invoke(stypy.reporting.localization.Localization(__file__, 160, 16), wrap_function_202766, *[fun_202767, args_202768], **kwargs_202769)
    
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___202771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 4), wrap_function_call_result_202770, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_202772 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), getitem___202771, int_202765)
    
    # Assigning a type to the variable 'tuple_var_assignment_202446' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'tuple_var_assignment_202446', subscript_call_result_202772)
    
    # Assigning a Name to a Name (line 160):
    # Getting the type of 'tuple_var_assignment_202445' (line 160)
    tuple_var_assignment_202445_202773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'tuple_var_assignment_202445')
    # Assigning a type to the variable 'nfun' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'nfun', tuple_var_assignment_202445_202773)
    
    # Assigning a Name to a Name (line 160):
    # Getting the type of 'tuple_var_assignment_202446' (line 160)
    tuple_var_assignment_202446_202774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'tuple_var_assignment_202446')
    # Assigning a type to the variable 'fun' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 10), 'fun', tuple_var_assignment_202446_202774)
    
    # Assigning a Call to a Tuple (line 161):
    
    # Assigning a Subscript to a Name (line 161):
    
    # Obtaining the type of the subscript
    int_202775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')
    
    # Call to wrap_function(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'jac' (line 161)
    jac_202777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 30), 'jac', False)
    # Getting the type of 'args' (line 161)
    args_202778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'args', False)
    # Processing the call keyword arguments (line 161)
    kwargs_202779 = {}
    # Getting the type of 'wrap_function' (line 161)
    wrap_function_202776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 161)
    wrap_function_call_result_202780 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), wrap_function_202776, *[jac_202777, args_202778], **kwargs_202779)
    
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___202781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), wrap_function_call_result_202780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_202782 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), getitem___202781, int_202775)
    
    # Assigning a type to the variable 'tuple_var_assignment_202447' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'tuple_var_assignment_202447', subscript_call_result_202782)
    
    # Assigning a Subscript to a Name (line 161):
    
    # Obtaining the type of the subscript
    int_202783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')
    
    # Call to wrap_function(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'jac' (line 161)
    jac_202785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 30), 'jac', False)
    # Getting the type of 'args' (line 161)
    args_202786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'args', False)
    # Processing the call keyword arguments (line 161)
    kwargs_202787 = {}
    # Getting the type of 'wrap_function' (line 161)
    wrap_function_202784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 161)
    wrap_function_call_result_202788 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), wrap_function_202784, *[jac_202785, args_202786], **kwargs_202787)
    
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___202789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), wrap_function_call_result_202788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_202790 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), getitem___202789, int_202783)
    
    # Assigning a type to the variable 'tuple_var_assignment_202448' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'tuple_var_assignment_202448', subscript_call_result_202790)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'tuple_var_assignment_202447' (line 161)
    tuple_var_assignment_202447_202791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'tuple_var_assignment_202447')
    # Assigning a type to the variable 'njac' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'njac', tuple_var_assignment_202447_202791)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'tuple_var_assignment_202448' (line 161)
    tuple_var_assignment_202448_202792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'tuple_var_assignment_202448')
    # Assigning a type to the variable 'jac' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 10), 'jac', tuple_var_assignment_202448_202792)
    
    # Assigning a Call to a Tuple (line 162):
    
    # Assigning a Subscript to a Name (line 162):
    
    # Obtaining the type of the subscript
    int_202793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 4), 'int')
    
    # Call to wrap_function(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'hess' (line 162)
    hess_202795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'hess', False)
    # Getting the type of 'args' (line 162)
    args_202796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 38), 'args', False)
    # Processing the call keyword arguments (line 162)
    kwargs_202797 = {}
    # Getting the type of 'wrap_function' (line 162)
    wrap_function_202794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 162)
    wrap_function_call_result_202798 = invoke(stypy.reporting.localization.Localization(__file__, 162, 18), wrap_function_202794, *[hess_202795, args_202796], **kwargs_202797)
    
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___202799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 4), wrap_function_call_result_202798, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_202800 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), getitem___202799, int_202793)
    
    # Assigning a type to the variable 'tuple_var_assignment_202449' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'tuple_var_assignment_202449', subscript_call_result_202800)
    
    # Assigning a Subscript to a Name (line 162):
    
    # Obtaining the type of the subscript
    int_202801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 4), 'int')
    
    # Call to wrap_function(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'hess' (line 162)
    hess_202803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'hess', False)
    # Getting the type of 'args' (line 162)
    args_202804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 38), 'args', False)
    # Processing the call keyword arguments (line 162)
    kwargs_202805 = {}
    # Getting the type of 'wrap_function' (line 162)
    wrap_function_202802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 162)
    wrap_function_call_result_202806 = invoke(stypy.reporting.localization.Localization(__file__, 162, 18), wrap_function_202802, *[hess_202803, args_202804], **kwargs_202805)
    
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___202807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 4), wrap_function_call_result_202806, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_202808 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), getitem___202807, int_202801)
    
    # Assigning a type to the variable 'tuple_var_assignment_202450' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'tuple_var_assignment_202450', subscript_call_result_202808)
    
    # Assigning a Name to a Name (line 162):
    # Getting the type of 'tuple_var_assignment_202449' (line 162)
    tuple_var_assignment_202449_202809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'tuple_var_assignment_202449')
    # Assigning a type to the variable 'nhess' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'nhess', tuple_var_assignment_202449_202809)
    
    # Assigning a Name to a Name (line 162):
    # Getting the type of 'tuple_var_assignment_202450' (line 162)
    tuple_var_assignment_202450_202810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'tuple_var_assignment_202450')
    # Assigning a type to the variable 'hess' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'hess', tuple_var_assignment_202450_202810)
    
    # Assigning a Call to a Tuple (line 163):
    
    # Assigning a Subscript to a Name (line 163):
    
    # Obtaining the type of the subscript
    int_202811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 4), 'int')
    
    # Call to wrap_function(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'hessp' (line 163)
    hessp_202813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'hessp', False)
    # Getting the type of 'args' (line 163)
    args_202814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'args', False)
    # Processing the call keyword arguments (line 163)
    kwargs_202815 = {}
    # Getting the type of 'wrap_function' (line 163)
    wrap_function_202812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 163)
    wrap_function_call_result_202816 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), wrap_function_202812, *[hessp_202813, args_202814], **kwargs_202815)
    
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___202817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 4), wrap_function_call_result_202816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_202818 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), getitem___202817, int_202811)
    
    # Assigning a type to the variable 'tuple_var_assignment_202451' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'tuple_var_assignment_202451', subscript_call_result_202818)
    
    # Assigning a Subscript to a Name (line 163):
    
    # Obtaining the type of the subscript
    int_202819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 4), 'int')
    
    # Call to wrap_function(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'hessp' (line 163)
    hessp_202821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'hessp', False)
    # Getting the type of 'args' (line 163)
    args_202822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'args', False)
    # Processing the call keyword arguments (line 163)
    kwargs_202823 = {}
    # Getting the type of 'wrap_function' (line 163)
    wrap_function_202820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 163)
    wrap_function_call_result_202824 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), wrap_function_202820, *[hessp_202821, args_202822], **kwargs_202823)
    
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___202825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 4), wrap_function_call_result_202824, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_202826 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), getitem___202825, int_202819)
    
    # Assigning a type to the variable 'tuple_var_assignment_202452' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'tuple_var_assignment_202452', subscript_call_result_202826)
    
    # Assigning a Name to a Name (line 163):
    # Getting the type of 'tuple_var_assignment_202451' (line 163)
    tuple_var_assignment_202451_202827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'tuple_var_assignment_202451')
    # Assigning a type to the variable 'nhessp' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'nhessp', tuple_var_assignment_202451_202827)
    
    # Assigning a Name to a Name (line 163):
    # Getting the type of 'tuple_var_assignment_202452' (line 163)
    tuple_var_assignment_202452_202828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'tuple_var_assignment_202452')
    # Assigning a type to the variable 'hessp' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'hessp', tuple_var_assignment_202452_202828)
    
    # Type idiom detected: calculating its left and rigth part (line 166)
    # Getting the type of 'maxiter' (line 166)
    maxiter_202829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 7), 'maxiter')
    # Getting the type of 'None' (line 166)
    None_202830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'None')
    
    (may_be_202831, more_types_in_union_202832) = may_be_none(maxiter_202829, None_202830)

    if may_be_202831:

        if more_types_in_union_202832:
            # Runtime conditional SSA (line 166)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 167):
        
        # Assigning a BinOp to a Name (line 167):
        
        # Call to len(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'x0' (line 167)
        x0_202834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'x0', False)
        # Processing the call keyword arguments (line 167)
        kwargs_202835 = {}
        # Getting the type of 'len' (line 167)
        len_202833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'len', False)
        # Calling len(args, kwargs) (line 167)
        len_call_result_202836 = invoke(stypy.reporting.localization.Localization(__file__, 167, 18), len_202833, *[x0_202834], **kwargs_202835)
        
        int_202837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 26), 'int')
        # Applying the binary operator '*' (line 167)
        result_mul_202838 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 18), '*', len_call_result_202836, int_202837)
        
        # Assigning a type to the variable 'maxiter' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'maxiter', result_mul_202838)

        if more_types_in_union_202832:
            # SSA join for if statement (line 166)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 170):
    
    # Assigning a Num to a Name (line 170):
    int_202839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 15), 'int')
    # Assigning a type to the variable 'warnflag' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'warnflag', int_202839)
    
    # Assigning a Name to a Name (line 173):
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'initial_trust_radius' (line 173)
    initial_trust_radius_202840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'initial_trust_radius')
    # Assigning a type to the variable 'trust_radius' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'trust_radius', initial_trust_radius_202840)
    
    # Assigning a Name to a Name (line 174):
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'x0' (line 174)
    x0_202841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'x0')
    # Assigning a type to the variable 'x' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'x', x0_202841)
    
    # Getting the type of 'return_all' (line 175)
    return_all_202842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 7), 'return_all')
    # Testing the type of an if condition (line 175)
    if_condition_202843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 4), return_all_202842)
    # Assigning a type to the variable 'if_condition_202843' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'if_condition_202843', if_condition_202843)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 176):
    
    # Assigning a List to a Name (line 176):
    
    # Obtaining an instance of the builtin type 'list' (line 176)
    list_202844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'x' (line 176)
    x_202845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_202844, x_202845)
    
    # Assigning a type to the variable 'allvecs' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'allvecs', list_202844)
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to subproblem(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'x' (line 177)
    x_202847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'x', False)
    # Getting the type of 'fun' (line 177)
    fun_202848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 22), 'fun', False)
    # Getting the type of 'jac' (line 177)
    jac_202849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), 'jac', False)
    # Getting the type of 'hess' (line 177)
    hess_202850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'hess', False)
    # Getting the type of 'hessp' (line 177)
    hessp_202851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 38), 'hessp', False)
    # Processing the call keyword arguments (line 177)
    kwargs_202852 = {}
    # Getting the type of 'subproblem' (line 177)
    subproblem_202846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'subproblem', False)
    # Calling subproblem(args, kwargs) (line 177)
    subproblem_call_result_202853 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), subproblem_202846, *[x_202847, fun_202848, jac_202849, hess_202850, hessp_202851], **kwargs_202852)
    
    # Assigning a type to the variable 'm' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'm', subproblem_call_result_202853)
    
    # Assigning a Num to a Name (line 178):
    
    # Assigning a Num to a Name (line 178):
    int_202854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 8), 'int')
    # Assigning a type to the variable 'k' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'k', int_202854)
    
    # Getting the type of 'True' (line 181)
    True_202855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 10), 'True')
    # Testing the type of an if condition (line 181)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), True_202855)
    # SSA begins for while statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # SSA begins for try-except statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 188):
    
    # Assigning a Subscript to a Name (line 188):
    
    # Obtaining the type of the subscript
    int_202856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'int')
    
    # Call to solve(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'trust_radius' (line 188)
    trust_radius_202859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 39), 'trust_radius', False)
    # Processing the call keyword arguments (line 188)
    kwargs_202860 = {}
    # Getting the type of 'm' (line 188)
    m_202857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'm', False)
    # Obtaining the member 'solve' of a type (line 188)
    solve_202858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 31), m_202857, 'solve')
    # Calling solve(args, kwargs) (line 188)
    solve_call_result_202861 = invoke(stypy.reporting.localization.Localization(__file__, 188, 31), solve_202858, *[trust_radius_202859], **kwargs_202860)
    
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___202862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), solve_call_result_202861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_202863 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), getitem___202862, int_202856)
    
    # Assigning a type to the variable 'tuple_var_assignment_202453' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_202453', subscript_call_result_202863)
    
    # Assigning a Subscript to a Name (line 188):
    
    # Obtaining the type of the subscript
    int_202864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'int')
    
    # Call to solve(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'trust_radius' (line 188)
    trust_radius_202867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 39), 'trust_radius', False)
    # Processing the call keyword arguments (line 188)
    kwargs_202868 = {}
    # Getting the type of 'm' (line 188)
    m_202865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'm', False)
    # Obtaining the member 'solve' of a type (line 188)
    solve_202866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 31), m_202865, 'solve')
    # Calling solve(args, kwargs) (line 188)
    solve_call_result_202869 = invoke(stypy.reporting.localization.Localization(__file__, 188, 31), solve_202866, *[trust_radius_202867], **kwargs_202868)
    
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___202870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), solve_call_result_202869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_202871 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), getitem___202870, int_202864)
    
    # Assigning a type to the variable 'tuple_var_assignment_202454' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_202454', subscript_call_result_202871)
    
    # Assigning a Name to a Name (line 188):
    # Getting the type of 'tuple_var_assignment_202453' (line 188)
    tuple_var_assignment_202453_202872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_202453')
    # Assigning a type to the variable 'p' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'p', tuple_var_assignment_202453_202872)
    
    # Assigning a Name to a Name (line 188):
    # Getting the type of 'tuple_var_assignment_202454' (line 188)
    tuple_var_assignment_202454_202873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'tuple_var_assignment_202454')
    # Assigning a type to the variable 'hits_boundary' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'hits_boundary', tuple_var_assignment_202454_202873)
    # SSA branch for the except part of a try statement (line 187)
    # SSA branch for the except 'Attribute' branch of a try statement (line 187)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'np' (line 189)
    np_202874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'np')
    # Obtaining the member 'linalg' of a type (line 189)
    linalg_202875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 15), np_202874, 'linalg')
    # Obtaining the member 'linalg' of a type (line 189)
    linalg_202876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 15), linalg_202875, 'linalg')
    # Obtaining the member 'LinAlgError' of a type (line 189)
    LinAlgError_202877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 15), linalg_202876, 'LinAlgError')
    # Assigning a type to the variable 'e' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'e', LinAlgError_202877)
    
    # Assigning a Num to a Name (line 190):
    
    # Assigning a Num to a Name (line 190):
    int_202878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 23), 'int')
    # Assigning a type to the variable 'warnflag' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'warnflag', int_202878)
    # SSA join for try-except statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to m(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'p' (line 194)
    p_202880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'p', False)
    # Processing the call keyword arguments (line 194)
    kwargs_202881 = {}
    # Getting the type of 'm' (line 194)
    m_202879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 26), 'm', False)
    # Calling m(args, kwargs) (line 194)
    m_call_result_202882 = invoke(stypy.reporting.localization.Localization(__file__, 194, 26), m_202879, *[p_202880], **kwargs_202881)
    
    # Assigning a type to the variable 'predicted_value' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'predicted_value', m_call_result_202882)
    
    # Assigning a BinOp to a Name (line 197):
    
    # Assigning a BinOp to a Name (line 197):
    # Getting the type of 'x' (line 197)
    x_202883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 21), 'x')
    # Getting the type of 'p' (line 197)
    p_202884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 25), 'p')
    # Applying the binary operator '+' (line 197)
    result_add_202885 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 21), '+', x_202883, p_202884)
    
    # Assigning a type to the variable 'x_proposed' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'x_proposed', result_add_202885)
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to subproblem(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'x_proposed' (line 198)
    x_proposed_202887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 32), 'x_proposed', False)
    # Getting the type of 'fun' (line 198)
    fun_202888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 44), 'fun', False)
    # Getting the type of 'jac' (line 198)
    jac_202889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 49), 'jac', False)
    # Getting the type of 'hess' (line 198)
    hess_202890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 54), 'hess', False)
    # Getting the type of 'hessp' (line 198)
    hessp_202891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 60), 'hessp', False)
    # Processing the call keyword arguments (line 198)
    kwargs_202892 = {}
    # Getting the type of 'subproblem' (line 198)
    subproblem_202886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'subproblem', False)
    # Calling subproblem(args, kwargs) (line 198)
    subproblem_call_result_202893 = invoke(stypy.reporting.localization.Localization(__file__, 198, 21), subproblem_202886, *[x_proposed_202887, fun_202888, jac_202889, hess_202890, hessp_202891], **kwargs_202892)
    
    # Assigning a type to the variable 'm_proposed' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'm_proposed', subproblem_call_result_202893)
    
    # Assigning a BinOp to a Name (line 201):
    
    # Assigning a BinOp to a Name (line 201):
    # Getting the type of 'm' (line 201)
    m_202894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'm')
    # Obtaining the member 'fun' of a type (line 201)
    fun_202895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 27), m_202894, 'fun')
    # Getting the type of 'm_proposed' (line 201)
    m_proposed_202896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 35), 'm_proposed')
    # Obtaining the member 'fun' of a type (line 201)
    fun_202897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 35), m_proposed_202896, 'fun')
    # Applying the binary operator '-' (line 201)
    result_sub_202898 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 27), '-', fun_202895, fun_202897)
    
    # Assigning a type to the variable 'actual_reduction' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'actual_reduction', result_sub_202898)
    
    # Assigning a BinOp to a Name (line 202):
    
    # Assigning a BinOp to a Name (line 202):
    # Getting the type of 'm' (line 202)
    m_202899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'm')
    # Obtaining the member 'fun' of a type (line 202)
    fun_202900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 30), m_202899, 'fun')
    # Getting the type of 'predicted_value' (line 202)
    predicted_value_202901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'predicted_value')
    # Applying the binary operator '-' (line 202)
    result_sub_202902 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 30), '-', fun_202900, predicted_value_202901)
    
    # Assigning a type to the variable 'predicted_reduction' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'predicted_reduction', result_sub_202902)
    
    
    # Getting the type of 'predicted_reduction' (line 203)
    predicted_reduction_202903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'predicted_reduction')
    int_202904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 34), 'int')
    # Applying the binary operator '<=' (line 203)
    result_le_202905 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), '<=', predicted_reduction_202903, int_202904)
    
    # Testing the type of an if condition (line 203)
    if_condition_202906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_le_202905)
    # Assigning a type to the variable 'if_condition_202906' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_202906', if_condition_202906)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 204):
    
    # Assigning a Num to a Name (line 204):
    int_202907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 23), 'int')
    # Assigning a type to the variable 'warnflag' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'warnflag', int_202907)
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 206):
    
    # Assigning a BinOp to a Name (line 206):
    # Getting the type of 'actual_reduction' (line 206)
    actual_reduction_202908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 14), 'actual_reduction')
    # Getting the type of 'predicted_reduction' (line 206)
    predicted_reduction_202909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 33), 'predicted_reduction')
    # Applying the binary operator 'div' (line 206)
    result_div_202910 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 14), 'div', actual_reduction_202908, predicted_reduction_202909)
    
    # Assigning a type to the variable 'rho' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'rho', result_div_202910)
    
    
    # Getting the type of 'rho' (line 209)
    rho_202911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'rho')
    float_202912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 17), 'float')
    # Applying the binary operator '<' (line 209)
    result_lt_202913 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), '<', rho_202911, float_202912)
    
    # Testing the type of an if condition (line 209)
    if_condition_202914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 8), result_lt_202913)
    # Assigning a type to the variable 'if_condition_202914' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'if_condition_202914', if_condition_202914)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'trust_radius' (line 210)
    trust_radius_202915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'trust_radius')
    float_202916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 28), 'float')
    # Applying the binary operator '*=' (line 210)
    result_imul_202917 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 12), '*=', trust_radius_202915, float_202916)
    # Assigning a type to the variable 'trust_radius' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'trust_radius', result_imul_202917)
    
    # SSA branch for the else part of an if statement (line 209)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rho' (line 211)
    rho_202918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'rho')
    float_202919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'float')
    # Applying the binary operator '>' (line 211)
    result_gt_202920 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 13), '>', rho_202918, float_202919)
    
    # Getting the type of 'hits_boundary' (line 211)
    hits_boundary_202921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 28), 'hits_boundary')
    # Applying the binary operator 'and' (line 211)
    result_and_keyword_202922 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 13), 'and', result_gt_202920, hits_boundary_202921)
    
    # Testing the type of an if condition (line 211)
    if_condition_202923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 13), result_and_keyword_202922)
    # Assigning a type to the variable 'if_condition_202923' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'if_condition_202923', if_condition_202923)
    # SSA begins for if statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to min(...): (line 212)
    # Processing the call arguments (line 212)
    int_202925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 31), 'int')
    # Getting the type of 'trust_radius' (line 212)
    trust_radius_202926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'trust_radius', False)
    # Applying the binary operator '*' (line 212)
    result_mul_202927 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 31), '*', int_202925, trust_radius_202926)
    
    # Getting the type of 'max_trust_radius' (line 212)
    max_trust_radius_202928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 47), 'max_trust_radius', False)
    # Processing the call keyword arguments (line 212)
    kwargs_202929 = {}
    # Getting the type of 'min' (line 212)
    min_202924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'min', False)
    # Calling min(args, kwargs) (line 212)
    min_call_result_202930 = invoke(stypy.reporting.localization.Localization(__file__, 212, 27), min_202924, *[result_mul_202927, max_trust_radius_202928], **kwargs_202929)
    
    # Assigning a type to the variable 'trust_radius' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'trust_radius', min_call_result_202930)
    # SSA join for if statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rho' (line 215)
    rho_202931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'rho')
    # Getting the type of 'eta' (line 215)
    eta_202932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'eta')
    # Applying the binary operator '>' (line 215)
    result_gt_202933 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), '>', rho_202931, eta_202932)
    
    # Testing the type of an if condition (line 215)
    if_condition_202934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_gt_202933)
    # Assigning a type to the variable 'if_condition_202934' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_202934', if_condition_202934)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 216):
    
    # Assigning a Name to a Name (line 216):
    # Getting the type of 'x_proposed' (line 216)
    x_proposed_202935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'x_proposed')
    # Assigning a type to the variable 'x' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'x', x_proposed_202935)
    
    # Assigning a Name to a Name (line 217):
    
    # Assigning a Name to a Name (line 217):
    # Getting the type of 'm_proposed' (line 217)
    m_proposed_202936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'm_proposed')
    # Assigning a type to the variable 'm' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'm', m_proposed_202936)
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_all' (line 220)
    return_all_202937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'return_all')
    # Testing the type of an if condition (line 220)
    if_condition_202938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 8), return_all_202937)
    # Assigning a type to the variable 'if_condition_202938' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'if_condition_202938', if_condition_202938)
    # SSA begins for if statement (line 220)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'x' (line 221)
    x_202941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 27), 'x', False)
    # Processing the call keyword arguments (line 221)
    kwargs_202942 = {}
    # Getting the type of 'allvecs' (line 221)
    allvecs_202939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'allvecs', False)
    # Obtaining the member 'append' of a type (line 221)
    append_202940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), allvecs_202939, 'append')
    # Calling append(args, kwargs) (line 221)
    append_call_result_202943 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), append_202940, *[x_202941], **kwargs_202942)
    
    # SSA join for if statement (line 220)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 222)
    # Getting the type of 'callback' (line 222)
    callback_202944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'callback')
    # Getting the type of 'None' (line 222)
    None_202945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 27), 'None')
    
    (may_be_202946, more_types_in_union_202947) = may_not_be_none(callback_202944, None_202945)

    if may_be_202946:

        if more_types_in_union_202947:
            # Runtime conditional SSA (line 222)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'x' (line 223)
        x_202949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'x', False)
        # Processing the call keyword arguments (line 223)
        kwargs_202950 = {}
        # Getting the type of 'callback' (line 223)
        callback_202948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'callback', False)
        # Calling callback(args, kwargs) (line 223)
        callback_call_result_202951 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), callback_202948, *[x_202949], **kwargs_202950)
        

        if more_types_in_union_202947:
            # SSA join for if statement (line 222)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'k' (line 224)
    k_202952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'k')
    int_202953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 13), 'int')
    # Applying the binary operator '+=' (line 224)
    result_iadd_202954 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 8), '+=', k_202952, int_202953)
    # Assigning a type to the variable 'k' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'k', result_iadd_202954)
    
    
    
    # Getting the type of 'm' (line 227)
    m_202955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'm')
    # Obtaining the member 'jac_mag' of a type (line 227)
    jac_mag_202956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), m_202955, 'jac_mag')
    # Getting the type of 'gtol' (line 227)
    gtol_202957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'gtol')
    # Applying the binary operator '<' (line 227)
    result_lt_202958 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), '<', jac_mag_202956, gtol_202957)
    
    # Testing the type of an if condition (line 227)
    if_condition_202959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_lt_202958)
    # Assigning a type to the variable 'if_condition_202959' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_202959', if_condition_202959)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 228):
    
    # Assigning a Num to a Name (line 228):
    int_202960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 23), 'int')
    # Assigning a type to the variable 'warnflag' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'warnflag', int_202960)
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 232)
    k_202961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'k')
    # Getting the type of 'maxiter' (line 232)
    maxiter_202962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'maxiter')
    # Applying the binary operator '>=' (line 232)
    result_ge_202963 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 11), '>=', k_202961, maxiter_202962)
    
    # Testing the type of an if condition (line 232)
    if_condition_202964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), result_ge_202963)
    # Assigning a type to the variable 'if_condition_202964' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_202964', if_condition_202964)
    # SSA begins for if statement (line 232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 233):
    
    # Assigning a Num to a Name (line 233):
    int_202965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 23), 'int')
    # Assigning a type to the variable 'warnflag' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'warnflag', int_202965)
    # SSA join for if statement (line 232)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 237):
    
    # Assigning a Tuple to a Name (line 237):
    
    # Obtaining an instance of the builtin type 'tuple' (line 238)
    tuple_202966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 238)
    # Adding element type (line 238)
    
    # Obtaining the type of the subscript
    str_202967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 28), 'str', 'success')
    # Getting the type of '_status_message' (line 238)
    _status_message_202968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), '_status_message')
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___202969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), _status_message_202968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_202970 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), getitem___202969, str_202967)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 12), tuple_202966, subscript_call_result_202970)
    # Adding element type (line 238)
    
    # Obtaining the type of the subscript
    str_202971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 28), 'str', 'maxiter')
    # Getting the type of '_status_message' (line 239)
    _status_message_202972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), '_status_message')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___202973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), _status_message_202972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_202974 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), getitem___202973, str_202971)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 12), tuple_202966, subscript_call_result_202974)
    # Adding element type (line 238)
    str_202975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 12), 'str', 'A bad approximation caused failure to predict improvement.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 12), tuple_202966, str_202975)
    # Adding element type (line 238)
    str_202976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 12), 'str', 'A linalg error occurred, such as a non-psd Hessian.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 12), tuple_202966, str_202976)
    
    # Assigning a type to the variable 'status_messages' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'status_messages', tuple_202966)
    
    # Getting the type of 'disp' (line 243)
    disp_202977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 7), 'disp')
    # Testing the type of an if condition (line 243)
    if_condition_202978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 4), disp_202977)
    # Assigning a type to the variable 'if_condition_202978' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'if_condition_202978', if_condition_202978)
    # SSA begins for if statement (line 243)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'warnflag' (line 244)
    warnflag_202979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 11), 'warnflag')
    int_202980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 23), 'int')
    # Applying the binary operator '==' (line 244)
    result_eq_202981 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 11), '==', warnflag_202979, int_202980)
    
    # Testing the type of an if condition (line 244)
    if_condition_202982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 8), result_eq_202981)
    # Assigning a type to the variable 'if_condition_202982' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'if_condition_202982', if_condition_202982)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining the type of the subscript
    # Getting the type of 'warnflag' (line 245)
    warnflag_202984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 34), 'warnflag', False)
    # Getting the type of 'status_messages' (line 245)
    status_messages_202985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 18), 'status_messages', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___202986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 18), status_messages_202985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_202987 = invoke(stypy.reporting.localization.Localization(__file__, 245, 18), getitem___202986, warnflag_202984)
    
    # Processing the call keyword arguments (line 245)
    kwargs_202988 = {}
    # Getting the type of 'print' (line 245)
    print_202983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'print', False)
    # Calling print(args, kwargs) (line 245)
    print_call_result_202989 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), print_202983, *[subscript_call_result_202987], **kwargs_202988)
    
    # SSA branch for the else part of an if statement (line 244)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 247)
    # Processing the call arguments (line 247)
    str_202991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 18), 'str', 'Warning: ')
    
    # Obtaining the type of the subscript
    # Getting the type of 'warnflag' (line 247)
    warnflag_202992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 48), 'warnflag', False)
    # Getting the type of 'status_messages' (line 247)
    status_messages_202993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 32), 'status_messages', False)
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___202994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 32), status_messages_202993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_202995 = invoke(stypy.reporting.localization.Localization(__file__, 247, 32), getitem___202994, warnflag_202992)
    
    # Applying the binary operator '+' (line 247)
    result_add_202996 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 18), '+', str_202991, subscript_call_result_202995)
    
    # Processing the call keyword arguments (line 247)
    kwargs_202997 = {}
    # Getting the type of 'print' (line 247)
    print_202990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'print', False)
    # Calling print(args, kwargs) (line 247)
    print_call_result_202998 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), print_202990, *[result_add_202996], **kwargs_202997)
    
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 248)
    # Processing the call arguments (line 248)
    str_203000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 14), 'str', '         Current function value: %f')
    # Getting the type of 'm' (line 248)
    m_203001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 54), 'm', False)
    # Obtaining the member 'fun' of a type (line 248)
    fun_203002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 54), m_203001, 'fun')
    # Applying the binary operator '%' (line 248)
    result_mod_203003 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 14), '%', str_203000, fun_203002)
    
    # Processing the call keyword arguments (line 248)
    kwargs_203004 = {}
    # Getting the type of 'print' (line 248)
    print_202999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'print', False)
    # Calling print(args, kwargs) (line 248)
    print_call_result_203005 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), print_202999, *[result_mod_203003], **kwargs_203004)
    
    
    # Call to print(...): (line 249)
    # Processing the call arguments (line 249)
    str_203007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 14), 'str', '         Iterations: %d')
    # Getting the type of 'k' (line 249)
    k_203008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'k', False)
    # Applying the binary operator '%' (line 249)
    result_mod_203009 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 14), '%', str_203007, k_203008)
    
    # Processing the call keyword arguments (line 249)
    kwargs_203010 = {}
    # Getting the type of 'print' (line 249)
    print_203006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'print', False)
    # Calling print(args, kwargs) (line 249)
    print_call_result_203011 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), print_203006, *[result_mod_203009], **kwargs_203010)
    
    
    # Call to print(...): (line 250)
    # Processing the call arguments (line 250)
    str_203013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 14), 'str', '         Function evaluations: %d')
    
    # Obtaining the type of the subscript
    int_203014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 57), 'int')
    # Getting the type of 'nfun' (line 250)
    nfun_203015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 52), 'nfun', False)
    # Obtaining the member '__getitem__' of a type (line 250)
    getitem___203016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 52), nfun_203015, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 250)
    subscript_call_result_203017 = invoke(stypy.reporting.localization.Localization(__file__, 250, 52), getitem___203016, int_203014)
    
    # Applying the binary operator '%' (line 250)
    result_mod_203018 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 14), '%', str_203013, subscript_call_result_203017)
    
    # Processing the call keyword arguments (line 250)
    kwargs_203019 = {}
    # Getting the type of 'print' (line 250)
    print_203012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'print', False)
    # Calling print(args, kwargs) (line 250)
    print_call_result_203020 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), print_203012, *[result_mod_203018], **kwargs_203019)
    
    
    # Call to print(...): (line 251)
    # Processing the call arguments (line 251)
    str_203022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 14), 'str', '         Gradient evaluations: %d')
    
    # Obtaining the type of the subscript
    int_203023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 57), 'int')
    # Getting the type of 'njac' (line 251)
    njac_203024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 52), 'njac', False)
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___203025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 52), njac_203024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_203026 = invoke(stypy.reporting.localization.Localization(__file__, 251, 52), getitem___203025, int_203023)
    
    # Applying the binary operator '%' (line 251)
    result_mod_203027 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 14), '%', str_203022, subscript_call_result_203026)
    
    # Processing the call keyword arguments (line 251)
    kwargs_203028 = {}
    # Getting the type of 'print' (line 251)
    print_203021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'print', False)
    # Calling print(args, kwargs) (line 251)
    print_call_result_203029 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), print_203021, *[result_mod_203027], **kwargs_203028)
    
    
    # Call to print(...): (line 252)
    # Processing the call arguments (line 252)
    str_203031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 14), 'str', '         Hessian evaluations: %d')
    
    # Obtaining the type of the subscript
    int_203032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 57), 'int')
    # Getting the type of 'nhess' (line 252)
    nhess_203033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 51), 'nhess', False)
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___203034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 51), nhess_203033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_203035 = invoke(stypy.reporting.localization.Localization(__file__, 252, 51), getitem___203034, int_203032)
    
    # Applying the binary operator '%' (line 252)
    result_mod_203036 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 14), '%', str_203031, subscript_call_result_203035)
    
    # Processing the call keyword arguments (line 252)
    kwargs_203037 = {}
    # Getting the type of 'print' (line 252)
    print_203030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'print', False)
    # Calling print(args, kwargs) (line 252)
    print_call_result_203038 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), print_203030, *[result_mod_203036], **kwargs_203037)
    
    # SSA join for if statement (line 243)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to OptimizeResult(...): (line 254)
    # Processing the call keyword arguments (line 254)
    # Getting the type of 'x' (line 254)
    x_203040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 30), 'x', False)
    keyword_203041 = x_203040
    
    # Getting the type of 'warnflag' (line 254)
    warnflag_203042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 42), 'warnflag', False)
    int_203043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 54), 'int')
    # Applying the binary operator '==' (line 254)
    result_eq_203044 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 42), '==', warnflag_203042, int_203043)
    
    keyword_203045 = result_eq_203044
    # Getting the type of 'warnflag' (line 254)
    warnflag_203046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 65), 'warnflag', False)
    keyword_203047 = warnflag_203046
    # Getting the type of 'm' (line 255)
    m_203048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 32), 'm', False)
    # Obtaining the member 'fun' of a type (line 255)
    fun_203049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 32), m_203048, 'fun')
    keyword_203050 = fun_203049
    # Getting the type of 'm' (line 255)
    m_203051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 43), 'm', False)
    # Obtaining the member 'jac' of a type (line 255)
    jac_203052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 43), m_203051, 'jac')
    keyword_203053 = jac_203052
    
    # Obtaining the type of the subscript
    int_203054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 60), 'int')
    # Getting the type of 'nfun' (line 255)
    nfun_203055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 55), 'nfun', False)
    # Obtaining the member '__getitem__' of a type (line 255)
    getitem___203056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 55), nfun_203055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 255)
    subscript_call_result_203057 = invoke(stypy.reporting.localization.Localization(__file__, 255, 55), getitem___203056, int_203054)
    
    keyword_203058 = subscript_call_result_203057
    
    # Obtaining the type of the subscript
    int_203059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 74), 'int')
    # Getting the type of 'njac' (line 255)
    njac_203060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 69), 'njac', False)
    # Obtaining the member '__getitem__' of a type (line 255)
    getitem___203061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 69), njac_203060, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 255)
    subscript_call_result_203062 = invoke(stypy.reporting.localization.Localization(__file__, 255, 69), getitem___203061, int_203059)
    
    keyword_203063 = subscript_call_result_203062
    
    # Obtaining the type of the subscript
    int_203064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 39), 'int')
    # Getting the type of 'nhess' (line 256)
    nhess_203065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 33), 'nhess', False)
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___203066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 33), nhess_203065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_203067 = invoke(stypy.reporting.localization.Localization(__file__, 256, 33), getitem___203066, int_203064)
    
    keyword_203068 = subscript_call_result_203067
    # Getting the type of 'k' (line 256)
    k_203069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 47), 'k', False)
    keyword_203070 = k_203069
    
    # Obtaining the type of the subscript
    # Getting the type of 'warnflag' (line 257)
    warnflag_203071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 52), 'warnflag', False)
    # Getting the type of 'status_messages' (line 257)
    status_messages_203072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 36), 'status_messages', False)
    # Obtaining the member '__getitem__' of a type (line 257)
    getitem___203073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 36), status_messages_203072, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 257)
    subscript_call_result_203074 = invoke(stypy.reporting.localization.Localization(__file__, 257, 36), getitem___203073, warnflag_203071)
    
    keyword_203075 = subscript_call_result_203074
    kwargs_203076 = {'status': keyword_203047, 'success': keyword_203045, 'njev': keyword_203063, 'nfev': keyword_203058, 'fun': keyword_203050, 'x': keyword_203041, 'message': keyword_203075, 'nhev': keyword_203068, 'jac': keyword_203053, 'nit': keyword_203070}
    # Getting the type of 'OptimizeResult' (line 254)
    OptimizeResult_203039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 13), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 254)
    OptimizeResult_call_result_203077 = invoke(stypy.reporting.localization.Localization(__file__, 254, 13), OptimizeResult_203039, *[], **kwargs_203076)
    
    # Assigning a type to the variable 'result' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'result', OptimizeResult_call_result_203077)
    
    # Type idiom detected: calculating its left and rigth part (line 259)
    # Getting the type of 'hess' (line 259)
    hess_203078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'hess')
    # Getting the type of 'None' (line 259)
    None_203079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 19), 'None')
    
    (may_be_203080, more_types_in_union_203081) = may_not_be_none(hess_203078, None_203079)

    if may_be_203080:

        if more_types_in_union_203081:
            # Runtime conditional SSA (line 259)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Subscript (line 260):
        
        # Assigning a Attribute to a Subscript (line 260):
        # Getting the type of 'm' (line 260)
        m_203082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'm')
        # Obtaining the member 'hess' of a type (line 260)
        hess_203083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 25), m_203082, 'hess')
        # Getting the type of 'result' (line 260)
        result_203084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'result')
        str_203085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 15), 'str', 'hess')
        # Storing an element on a container (line 260)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), result_203084, (str_203085, hess_203083))

        if more_types_in_union_203081:
            # SSA join for if statement (line 259)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'return_all' (line 262)
    return_all_203086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 7), 'return_all')
    # Testing the type of an if condition (line 262)
    if_condition_203087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 4), return_all_203086)
    # Assigning a type to the variable 'if_condition_203087' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'if_condition_203087', if_condition_203087)
    # SSA begins for if statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 263):
    
    # Assigning a Name to a Subscript (line 263):
    # Getting the type of 'allvecs' (line 263)
    allvecs_203088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 28), 'allvecs')
    # Getting the type of 'result' (line 263)
    result_203089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'result')
    str_203090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 15), 'str', 'allvecs')
    # Storing an element on a container (line 263)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), result_203089, (str_203090, allvecs_203088))
    # SSA join for if statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 265)
    result_203091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type', result_203091)
    
    # ################# End of '_minimize_trust_region(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minimize_trust_region' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_203092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_203092)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minimize_trust_region'
    return stypy_return_type_203092

# Assigning a type to the variable '_minimize_trust_region' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), '_minimize_trust_region', _minimize_trust_region)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
