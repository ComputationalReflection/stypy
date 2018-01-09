
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: import numpy as np
3: 
4: 
5: def check_arguments(fun, y0, support_complex):
6:     '''Helper function for checking arguments common to all solvers.'''
7:     y0 = np.asarray(y0)
8:     if np.issubdtype(y0.dtype, np.complexfloating):
9:         if not support_complex:
10:             raise ValueError("`y0` is complex, but the chosen solver does "
11:                              "not support integration in a complex domain.")
12:         dtype = complex
13:     else:
14:         dtype = float
15:     y0 = y0.astype(dtype, copy=False)
16: 
17:     if y0.ndim != 1:
18:         raise ValueError("`y0` must be 1-dimensional.")
19: 
20:     def fun_wrapped(t, y):
21:         return np.asarray(fun(t, y), dtype=dtype)
22: 
23:     return fun_wrapped, y0
24: 
25: 
26: class OdeSolver(object):
27:     '''Base class for ODE solvers.
28: 
29:     In order to implement a new solver you need to follow the guidelines:
30: 
31:         1. A constructor must accept parameters presented in the base class
32:            (listed below) along with any other parameters specific to a solver.
33:         2. A constructor must accept arbitrary extraneous arguments
34:            ``**extraneous``, but warn that these arguments are irrelevant
35:            using `common.warn_extraneous` function. Do not pass these
36:            arguments to the base class.
37:         3. A solver must implement a private method `_step_impl(self)` which
38:            propagates a solver one step further. It must return tuple
39:            ``(success, message)``, where ``success`` is a boolean indicating
40:            whether a step was successful, and ``message`` is a string
41:            containing description of a failure if a step failed or None
42:            otherwise.
43:         4. A solver must implement a private method `_dense_output_impl(self)`
44:            which returns a `DenseOutput` object covering the last successful
45:            step.
46:         5. A solver must have attributes listed below in Attributes section.
47:            Note that `t_old` and `step_size` are updated automatically.
48:         6. Use `fun(self, t, y)` method for the system rhs evaluation, this
49:            way the number of function evaluations (`nfev`) will be tracked
50:            automatically.
51:         7. For convenience a base class provides `fun_single(self, t, y)` and
52:            `fun_vectorized(self, t, y)` for evaluating the rhs in
53:            non-vectorized and vectorized fashions respectively (regardless of
54:            how `fun` from the constructor is implemented). These calls don't
55:            increment `nfev`.
56:         8. If a solver uses a Jacobian matrix and LU decompositions, it should
57:            track the number of Jacobian evaluations (`njev`) and the number of
58:            LU decompositions (`nlu`).
59:         9. By convention the function evaluations used to compute a finite
60:            difference approximation of the Jacobian should not be counted in
61:            `nfev`, thus use `fun_single(self, t, y)` or
62:            `fun_vectorized(self, t, y)` when computing a finite difference
63:            approximation of the Jacobian.
64: 
65:     Parameters
66:     ----------
67:     fun : callable
68:         Right-hand side of the system. The calling signature is ``fun(t, y)``.
69:         Here ``t`` is a scalar and there are two options for ndarray ``y``.
70:         It can either have shape (n,), then ``fun`` must return array_like with
71:         shape (n,). Or alternatively it can have shape (n, n_points), then
72:         ``fun`` must return array_like with shape (n, n_points) (each column
73:         corresponds to a single column in ``y``). The choice between the two
74:         options is determined by `vectorized` argument (see below).
75:     t0 : float
76:         Initial time.
77:     y0 : array_like, shape (n,)
78:         Initial state.
79:     t_bound : float
80:         Boundary time --- the integration won't continue beyond it. It also
81:         determines the direction of the integration.
82:     vectorized : bool
83:         Whether `fun` is implemented in a vectorized fashion.
84:     support_complex : bool, optional
85:         Whether integration in a complex domain should be supported.
86:         Generally determined by a derived solver class capabilities.
87:         Default is False.
88: 
89:     Attributes
90:     ----------
91:     n : int
92:         Number of equations.
93:     status : string
94:         Current status of the solver: 'running', 'finished' or 'failed'.
95:     t_bound : float
96:         Boundary time.
97:     direction : float
98:         Integration direction: +1 or -1.
99:     t : float
100:         Current time.
101:     y : ndarray
102:         Current state.
103:     t_old : float
104:         Previous time. None if no steps were made yet.
105:     step_size : float
106:         Size of the last successful step. None if no steps were made yet.
107:     nfev : int
108:         Number of the system's rhs evaluations.
109:     njev : int
110:         Number of the Jacobian evaluations.
111:     nlu : int
112:         Number of LU decompositions.
113:     '''
114:     TOO_SMALL_STEP = "Required step size is less than spacing between numbers."
115: 
116:     def __init__(self, fun, t0, y0, t_bound, vectorized,
117:                  support_complex=False):
118:         self.t_old = None
119:         self.t = t0
120:         self._fun, self.y = check_arguments(fun, y0, support_complex)
121:         self.t_bound = t_bound
122:         self.vectorized = vectorized
123: 
124:         if vectorized:
125:             def fun_single(t, y):
126:                 return self._fun(t, y[:, None]).ravel()
127:             fun_vectorized = self._fun
128:         else:
129:             fun_single = self._fun
130: 
131:             def fun_vectorized(t, y):
132:                 f = np.empty_like(y)
133:                 for i, yi in enumerate(y.T):
134:                     f[:, i] = self._fun(t, yi)
135:                 return f
136: 
137:         def fun(t, y):
138:             self.nfev += 1
139:             return self.fun_single(t, y)
140: 
141:         self.fun = fun
142:         self.fun_single = fun_single
143:         self.fun_vectorized = fun_vectorized
144: 
145:         self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
146:         self.n = self.y.size
147:         self.status = 'running'
148: 
149:         self.nfev = 0
150:         self.njev = 0
151:         self.nlu = 0
152: 
153:     @property
154:     def step_size(self):
155:         if self.t_old is None:
156:             return None
157:         else:
158:             return np.abs(self.t - self.t_old)
159: 
160:     def step(self):
161:         '''Perform one integration step.
162: 
163:         Returns
164:         -------
165:         message : string or None
166:             Report from the solver. Typically a reason for a failure if
167:             `self.status` is 'failed' after the step was taken or None
168:             otherwise.
169:         '''
170:         if self.status != 'running':
171:             raise RuntimeError("Attempt to step on a failed or finished "
172:                                "solver.")
173: 
174:         if self.n == 0 or self.t == self.t_bound:
175:             # Handle corner cases of empty solver or no integration.
176:             self.t_old = self.t
177:             self.t = self.t_bound
178:             message = None
179:             self.status = 'finished'
180:         else:
181:             t = self.t
182:             success, message = self._step_impl()
183: 
184:             if not success:
185:                 self.status = 'failed'
186:             else:
187:                 self.t_old = t
188:                 if self.direction * (self.t - self.t_bound) >= 0:
189:                     self.status = 'finished'
190: 
191:         return message
192: 
193:     def dense_output(self):
194:         '''Compute a local interpolant over the last successful step.
195: 
196:         Returns
197:         -------
198:         sol : `DenseOutput`
199:             Local interpolant over the last successful step.
200:         '''
201:         if self.t_old is None:
202:             raise RuntimeError("Dense output is available after a successful "
203:                                "step was made.")
204: 
205:         if self.n == 0 or self.t == self.t_old:
206:             # Handle corner cases of empty solver and no integration.
207:             return ConstantDenseOutput(self.t_old, self.t, self.y)
208:         else:
209:             return self._dense_output_impl()
210: 
211:     def _step_impl(self):
212:         raise NotImplementedError
213: 
214:     def _dense_output_impl(self):
215:         raise NotImplementedError
216: 
217: 
218: class DenseOutput(object):
219:     '''Base class for local interpolant over step made by an ODE solver.
220: 
221:     It interpolates between `t_min` and `t_max` (see Attributes below).
222:     Evaluation outside this interval is not forbidden, but the accuracy is not
223:     guaranteed.
224: 
225:     Attributes
226:     ----------
227:     t_min, t_max : float
228:         Time range of the interpolation.
229:     '''
230:     def __init__(self, t_old, t):
231:         self.t_old = t_old
232:         self.t = t
233:         self.t_min = min(t, t_old)
234:         self.t_max = max(t, t_old)
235: 
236:     def __call__(self, t):
237:         '''Evaluate the interpolant.
238: 
239:         Parameters
240:         ----------
241:         t : float or array_like with shape (n_points,)
242:             Points to evaluate the solution at.
243: 
244:         Returns
245:         -------
246:         y : ndarray, shape (n,) or (n, n_points)
247:             Computed values. Shape depends on whether `t` was a scalar or a
248:             1-d array.
249:         '''
250:         t = np.asarray(t)
251:         if t.ndim > 1:
252:             raise ValueError("`t` must be float or 1-d array.")
253:         return self._call_impl(t)
254: 
255:     def _call_impl(self, t):
256:         raise NotImplementedError
257: 
258: 
259: class ConstantDenseOutput(DenseOutput):
260:     '''Constant value interpolator.
261: 
262:     This class used for degenerate integration cases: equal integration limits
263:     or a system with 0 equations.
264:     '''
265:     def __init__(self, t_old, t, value):
266:         super(ConstantDenseOutput, self).__init__(t_old, t)
267:         self.value = value
268: 
269:     def _call_impl(self, t):
270:         if t.ndim == 0:
271:             return self.value
272:         else:
273:             ret = np.empty((self.value.shape[0], t.shape[0]))
274:             ret[:] = self.value[:, None]
275:             return ret
276: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_52188 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_52188) is not StypyTypeError):

    if (import_52188 != 'pyd_module'):
        __import__(import_52188)
        sys_modules_52189 = sys.modules[import_52188]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_52189.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_52188)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')


@norecursion
def check_arguments(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_arguments'
    module_type_store = module_type_store.open_function_context('check_arguments', 5, 0, False)
    
    # Passed parameters checking function
    check_arguments.stypy_localization = localization
    check_arguments.stypy_type_of_self = None
    check_arguments.stypy_type_store = module_type_store
    check_arguments.stypy_function_name = 'check_arguments'
    check_arguments.stypy_param_names_list = ['fun', 'y0', 'support_complex']
    check_arguments.stypy_varargs_param_name = None
    check_arguments.stypy_kwargs_param_name = None
    check_arguments.stypy_call_defaults = defaults
    check_arguments.stypy_call_varargs = varargs
    check_arguments.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_arguments', ['fun', 'y0', 'support_complex'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_arguments', localization, ['fun', 'y0', 'support_complex'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_arguments(...)' code ##################

    str_52190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'str', 'Helper function for checking arguments common to all solvers.')
    
    # Assigning a Call to a Name (line 7):
    
    # Assigning a Call to a Name (line 7):
    
    # Call to asarray(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'y0' (line 7)
    y0_52193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'y0', False)
    # Processing the call keyword arguments (line 7)
    kwargs_52194 = {}
    # Getting the type of 'np' (line 7)
    np_52191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'np', False)
    # Obtaining the member 'asarray' of a type (line 7)
    asarray_52192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), np_52191, 'asarray')
    # Calling asarray(args, kwargs) (line 7)
    asarray_call_result_52195 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), asarray_52192, *[y0_52193], **kwargs_52194)
    
    # Assigning a type to the variable 'y0' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'y0', asarray_call_result_52195)
    
    
    # Call to issubdtype(...): (line 8)
    # Processing the call arguments (line 8)
    # Getting the type of 'y0' (line 8)
    y0_52198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 21), 'y0', False)
    # Obtaining the member 'dtype' of a type (line 8)
    dtype_52199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 21), y0_52198, 'dtype')
    # Getting the type of 'np' (line 8)
    np_52200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 31), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 8)
    complexfloating_52201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 31), np_52200, 'complexfloating')
    # Processing the call keyword arguments (line 8)
    kwargs_52202 = {}
    # Getting the type of 'np' (line 8)
    np_52196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 8)
    issubdtype_52197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 7), np_52196, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 8)
    issubdtype_call_result_52203 = invoke(stypy.reporting.localization.Localization(__file__, 8, 7), issubdtype_52197, *[dtype_52199, complexfloating_52201], **kwargs_52202)
    
    # Testing the type of an if condition (line 8)
    if_condition_52204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 4), issubdtype_call_result_52203)
    # Assigning a type to the variable 'if_condition_52204' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'if_condition_52204', if_condition_52204)
    # SSA begins for if statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'support_complex' (line 9)
    support_complex_52205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'support_complex')
    # Applying the 'not' unary operator (line 9)
    result_not__52206 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 11), 'not', support_complex_52205)
    
    # Testing the type of an if condition (line 9)
    if_condition_52207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 8), result_not__52206)
    # Assigning a type to the variable 'if_condition_52207' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'if_condition_52207', if_condition_52207)
    # SSA begins for if statement (line 9)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 10)
    # Processing the call arguments (line 10)
    str_52209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 29), 'str', '`y0` is complex, but the chosen solver does not support integration in a complex domain.')
    # Processing the call keyword arguments (line 10)
    kwargs_52210 = {}
    # Getting the type of 'ValueError' (line 10)
    ValueError_52208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 10)
    ValueError_call_result_52211 = invoke(stypy.reporting.localization.Localization(__file__, 10, 18), ValueError_52208, *[str_52209], **kwargs_52210)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 10, 12), ValueError_call_result_52211, 'raise parameter', BaseException)
    # SSA join for if statement (line 9)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 12):
    
    # Assigning a Name to a Name (line 12):
    # Getting the type of 'complex' (line 12)
    complex_52212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'complex')
    # Assigning a type to the variable 'dtype' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'dtype', complex_52212)
    # SSA branch for the else part of an if statement (line 8)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 14):
    
    # Assigning a Name to a Name (line 14):
    # Getting the type of 'float' (line 14)
    float_52213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'float')
    # Assigning a type to the variable 'dtype' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'dtype', float_52213)
    # SSA join for if statement (line 8)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 15):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to astype(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'dtype' (line 15)
    dtype_52216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'dtype', False)
    # Processing the call keyword arguments (line 15)
    # Getting the type of 'False' (line 15)
    False_52217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 31), 'False', False)
    keyword_52218 = False_52217
    kwargs_52219 = {'copy': keyword_52218}
    # Getting the type of 'y0' (line 15)
    y0_52214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'y0', False)
    # Obtaining the member 'astype' of a type (line 15)
    astype_52215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 9), y0_52214, 'astype')
    # Calling astype(args, kwargs) (line 15)
    astype_call_result_52220 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), astype_52215, *[dtype_52216], **kwargs_52219)
    
    # Assigning a type to the variable 'y0' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'y0', astype_call_result_52220)
    
    
    # Getting the type of 'y0' (line 17)
    y0_52221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'y0')
    # Obtaining the member 'ndim' of a type (line 17)
    ndim_52222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 7), y0_52221, 'ndim')
    int_52223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'int')
    # Applying the binary operator '!=' (line 17)
    result_ne_52224 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 7), '!=', ndim_52222, int_52223)
    
    # Testing the type of an if condition (line 17)
    if_condition_52225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), result_ne_52224)
    # Assigning a type to the variable 'if_condition_52225' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_52225', if_condition_52225)
    # SSA begins for if statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 18)
    # Processing the call arguments (line 18)
    str_52227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', '`y0` must be 1-dimensional.')
    # Processing the call keyword arguments (line 18)
    kwargs_52228 = {}
    # Getting the type of 'ValueError' (line 18)
    ValueError_52226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 18)
    ValueError_call_result_52229 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), ValueError_52226, *[str_52227], **kwargs_52228)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 18, 8), ValueError_call_result_52229, 'raise parameter', BaseException)
    # SSA join for if statement (line 17)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def fun_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_wrapped'
        module_type_store = module_type_store.open_function_context('fun_wrapped', 20, 4, False)
        
        # Passed parameters checking function
        fun_wrapped.stypy_localization = localization
        fun_wrapped.stypy_type_of_self = None
        fun_wrapped.stypy_type_store = module_type_store
        fun_wrapped.stypy_function_name = 'fun_wrapped'
        fun_wrapped.stypy_param_names_list = ['t', 'y']
        fun_wrapped.stypy_varargs_param_name = None
        fun_wrapped.stypy_kwargs_param_name = None
        fun_wrapped.stypy_call_defaults = defaults
        fun_wrapped.stypy_call_varargs = varargs
        fun_wrapped.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun_wrapped', ['t', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_wrapped', localization, ['t', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_wrapped(...)' code ##################

        
        # Call to asarray(...): (line 21)
        # Processing the call arguments (line 21)
        
        # Call to fun(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 't' (line 21)
        t_52233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 30), 't', False)
        # Getting the type of 'y' (line 21)
        y_52234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 33), 'y', False)
        # Processing the call keyword arguments (line 21)
        kwargs_52235 = {}
        # Getting the type of 'fun' (line 21)
        fun_52232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'fun', False)
        # Calling fun(args, kwargs) (line 21)
        fun_call_result_52236 = invoke(stypy.reporting.localization.Localization(__file__, 21, 26), fun_52232, *[t_52233, y_52234], **kwargs_52235)
        
        # Processing the call keyword arguments (line 21)
        # Getting the type of 'dtype' (line 21)
        dtype_52237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 43), 'dtype', False)
        keyword_52238 = dtype_52237
        kwargs_52239 = {'dtype': keyword_52238}
        # Getting the type of 'np' (line 21)
        np_52230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'np', False)
        # Obtaining the member 'asarray' of a type (line 21)
        asarray_52231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 15), np_52230, 'asarray')
        # Calling asarray(args, kwargs) (line 21)
        asarray_call_result_52240 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), asarray_52231, *[fun_call_result_52236], **kwargs_52239)
        
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', asarray_call_result_52240)
        
        # ################# End of 'fun_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_52241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52241)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_wrapped'
        return stypy_return_type_52241

    # Assigning a type to the variable 'fun_wrapped' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'fun_wrapped', fun_wrapped)
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_52242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 'fun_wrapped' (line 23)
    fun_wrapped_52243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'fun_wrapped')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_52242, fun_wrapped_52243)
    # Adding element type (line 23)
    # Getting the type of 'y0' (line 23)
    y0_52244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'y0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_52242, y0_52244)
    
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', tuple_52242)
    
    # ################# End of 'check_arguments(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_arguments' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_52245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52245)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_arguments'
    return stypy_return_type_52245

# Assigning a type to the variable 'check_arguments' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'check_arguments', check_arguments)
# Declaration of the 'OdeSolver' class

class OdeSolver(object, ):
    str_52246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, (-1)), 'str', "Base class for ODE solvers.\n\n    In order to implement a new solver you need to follow the guidelines:\n\n        1. A constructor must accept parameters presented in the base class\n           (listed below) along with any other parameters specific to a solver.\n        2. A constructor must accept arbitrary extraneous arguments\n           ``**extraneous``, but warn that these arguments are irrelevant\n           using `common.warn_extraneous` function. Do not pass these\n           arguments to the base class.\n        3. A solver must implement a private method `_step_impl(self)` which\n           propagates a solver one step further. It must return tuple\n           ``(success, message)``, where ``success`` is a boolean indicating\n           whether a step was successful, and ``message`` is a string\n           containing description of a failure if a step failed or None\n           otherwise.\n        4. A solver must implement a private method `_dense_output_impl(self)`\n           which returns a `DenseOutput` object covering the last successful\n           step.\n        5. A solver must have attributes listed below in Attributes section.\n           Note that `t_old` and `step_size` are updated automatically.\n        6. Use `fun(self, t, y)` method for the system rhs evaluation, this\n           way the number of function evaluations (`nfev`) will be tracked\n           automatically.\n        7. For convenience a base class provides `fun_single(self, t, y)` and\n           `fun_vectorized(self, t, y)` for evaluating the rhs in\n           non-vectorized and vectorized fashions respectively (regardless of\n           how `fun` from the constructor is implemented). These calls don't\n           increment `nfev`.\n        8. If a solver uses a Jacobian matrix and LU decompositions, it should\n           track the number of Jacobian evaluations (`njev`) and the number of\n           LU decompositions (`nlu`).\n        9. By convention the function evaluations used to compute a finite\n           difference approximation of the Jacobian should not be counted in\n           `nfev`, thus use `fun_single(self, t, y)` or\n           `fun_vectorized(self, t, y)` when computing a finite difference\n           approximation of the Jacobian.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(t, y)``.\n        Here ``t`` is a scalar and there are two options for ndarray ``y``.\n        It can either have shape (n,), then ``fun`` must return array_like with\n        shape (n,). Or alternatively it can have shape (n, n_points), then\n        ``fun`` must return array_like with shape (n, n_points) (each column\n        corresponds to a single column in ``y``). The choice between the two\n        options is determined by `vectorized` argument (see below).\n    t0 : float\n        Initial time.\n    y0 : array_like, shape (n,)\n        Initial state.\n    t_bound : float\n        Boundary time --- the integration won't continue beyond it. It also\n        determines the direction of the integration.\n    vectorized : bool\n        Whether `fun` is implemented in a vectorized fashion.\n    support_complex : bool, optional\n        Whether integration in a complex domain should be supported.\n        Generally determined by a derived solver class capabilities.\n        Default is False.\n\n    Attributes\n    ----------\n    n : int\n        Number of equations.\n    status : string\n        Current status of the solver: 'running', 'finished' or 'failed'.\n    t_bound : float\n        Boundary time.\n    direction : float\n        Integration direction: +1 or -1.\n    t : float\n        Current time.\n    y : ndarray\n        Current state.\n    t_old : float\n        Previous time. None if no steps were made yet.\n    step_size : float\n        Size of the last successful step. None if no steps were made yet.\n    nfev : int\n        Number of the system's rhs evaluations.\n    njev : int\n        Number of the Jacobian evaluations.\n    nlu : int\n        Number of LU decompositions.\n    ")
    
    # Assigning a Str to a Name (line 114):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 117)
        False_52247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'False')
        defaults = [False_52247]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolver.__init__', ['fun', 't0', 'y0', 't_bound', 'vectorized', 'support_complex'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fun', 't0', 'y0', 't_bound', 'vectorized', 'support_complex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 118):
        
        # Assigning a Name to a Attribute (line 118):
        # Getting the type of 'None' (line 118)
        None_52248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'None')
        # Getting the type of 'self' (line 118)
        self_52249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self')
        # Setting the type of the member 't_old' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_52249, 't_old', None_52248)
        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 't0' (line 119)
        t0_52250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 't0')
        # Getting the type of 'self' (line 119)
        self_52251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 't' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_52251, 't', t0_52250)
        
        # Assigning a Call to a Tuple (line 120):
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_52252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'int')
        
        # Call to check_arguments(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'fun' (line 120)
        fun_52254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), 'fun', False)
        # Getting the type of 'y0' (line 120)
        y0_52255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 49), 'y0', False)
        # Getting the type of 'support_complex' (line 120)
        support_complex_52256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 53), 'support_complex', False)
        # Processing the call keyword arguments (line 120)
        kwargs_52257 = {}
        # Getting the type of 'check_arguments' (line 120)
        check_arguments_52253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'check_arguments', False)
        # Calling check_arguments(args, kwargs) (line 120)
        check_arguments_call_result_52258 = invoke(stypy.reporting.localization.Localization(__file__, 120, 28), check_arguments_52253, *[fun_52254, y0_52255, support_complex_52256], **kwargs_52257)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___52259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), check_arguments_call_result_52258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_52260 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), getitem___52259, int_52252)
        
        # Assigning a type to the variable 'tuple_var_assignment_52184' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_52184', subscript_call_result_52260)
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_52261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'int')
        
        # Call to check_arguments(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'fun' (line 120)
        fun_52263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), 'fun', False)
        # Getting the type of 'y0' (line 120)
        y0_52264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 49), 'y0', False)
        # Getting the type of 'support_complex' (line 120)
        support_complex_52265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 53), 'support_complex', False)
        # Processing the call keyword arguments (line 120)
        kwargs_52266 = {}
        # Getting the type of 'check_arguments' (line 120)
        check_arguments_52262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'check_arguments', False)
        # Calling check_arguments(args, kwargs) (line 120)
        check_arguments_call_result_52267 = invoke(stypy.reporting.localization.Localization(__file__, 120, 28), check_arguments_52262, *[fun_52263, y0_52264, support_complex_52265], **kwargs_52266)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___52268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), check_arguments_call_result_52267, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_52269 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), getitem___52268, int_52261)
        
        # Assigning a type to the variable 'tuple_var_assignment_52185' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_52185', subscript_call_result_52269)
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'tuple_var_assignment_52184' (line 120)
        tuple_var_assignment_52184_52270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_52184')
        # Getting the type of 'self' (line 120)
        self_52271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member '_fun' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_52271, '_fun', tuple_var_assignment_52184_52270)
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'tuple_var_assignment_52185' (line 120)
        tuple_var_assignment_52185_52272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_52185')
        # Getting the type of 'self' (line 120)
        self_52273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'self')
        # Setting the type of the member 'y' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 19), self_52273, 'y', tuple_var_assignment_52185_52272)
        
        # Assigning a Name to a Attribute (line 121):
        
        # Assigning a Name to a Attribute (line 121):
        # Getting the type of 't_bound' (line 121)
        t_bound_52274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 't_bound')
        # Getting the type of 'self' (line 121)
        self_52275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 't_bound' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_52275, 't_bound', t_bound_52274)
        
        # Assigning a Name to a Attribute (line 122):
        
        # Assigning a Name to a Attribute (line 122):
        # Getting the type of 'vectorized' (line 122)
        vectorized_52276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 26), 'vectorized')
        # Getting the type of 'self' (line 122)
        self_52277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member 'vectorized' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_52277, 'vectorized', vectorized_52276)
        
        # Getting the type of 'vectorized' (line 124)
        vectorized_52278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'vectorized')
        # Testing the type of an if condition (line 124)
        if_condition_52279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), vectorized_52278)
        # Assigning a type to the variable 'if_condition_52279' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_52279', if_condition_52279)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def fun_single(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fun_single'
            module_type_store = module_type_store.open_function_context('fun_single', 125, 12, False)
            
            # Passed parameters checking function
            fun_single.stypy_localization = localization
            fun_single.stypy_type_of_self = None
            fun_single.stypy_type_store = module_type_store
            fun_single.stypy_function_name = 'fun_single'
            fun_single.stypy_param_names_list = ['t', 'y']
            fun_single.stypy_varargs_param_name = None
            fun_single.stypy_kwargs_param_name = None
            fun_single.stypy_call_defaults = defaults
            fun_single.stypy_call_varargs = varargs
            fun_single.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fun_single', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fun_single', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fun_single(...)' code ##################

            
            # Call to ravel(...): (line 126)
            # Processing the call keyword arguments (line 126)
            kwargs_52291 = {}
            
            # Call to _fun(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 't' (line 126)
            t_52282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 't', False)
            
            # Obtaining the type of the subscript
            slice_52283 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 36), None, None, None)
            # Getting the type of 'None' (line 126)
            None_52284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'None', False)
            # Getting the type of 'y' (line 126)
            y_52285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 126)
            getitem___52286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 36), y_52285, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 126)
            subscript_call_result_52287 = invoke(stypy.reporting.localization.Localization(__file__, 126, 36), getitem___52286, (slice_52283, None_52284))
            
            # Processing the call keyword arguments (line 126)
            kwargs_52288 = {}
            # Getting the type of 'self' (line 126)
            self_52280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'self', False)
            # Obtaining the member '_fun' of a type (line 126)
            _fun_52281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), self_52280, '_fun')
            # Calling _fun(args, kwargs) (line 126)
            _fun_call_result_52289 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), _fun_52281, *[t_52282, subscript_call_result_52287], **kwargs_52288)
            
            # Obtaining the member 'ravel' of a type (line 126)
            ravel_52290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), _fun_call_result_52289, 'ravel')
            # Calling ravel(args, kwargs) (line 126)
            ravel_call_result_52292 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), ravel_52290, *[], **kwargs_52291)
            
            # Assigning a type to the variable 'stypy_return_type' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'stypy_return_type', ravel_call_result_52292)
            
            # ################# End of 'fun_single(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fun_single' in the type store
            # Getting the type of 'stypy_return_type' (line 125)
            stypy_return_type_52293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_52293)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fun_single'
            return stypy_return_type_52293

        # Assigning a type to the variable 'fun_single' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'fun_single', fun_single)
        
        # Assigning a Attribute to a Name (line 127):
        
        # Assigning a Attribute to a Name (line 127):
        # Getting the type of 'self' (line 127)
        self_52294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'self')
        # Obtaining the member '_fun' of a type (line 127)
        _fun_52295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 29), self_52294, '_fun')
        # Assigning a type to the variable 'fun_vectorized' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'fun_vectorized', _fun_52295)
        # SSA branch for the else part of an if statement (line 124)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 129):
        
        # Assigning a Attribute to a Name (line 129):
        # Getting the type of 'self' (line 129)
        self_52296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'self')
        # Obtaining the member '_fun' of a type (line 129)
        _fun_52297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), self_52296, '_fun')
        # Assigning a type to the variable 'fun_single' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'fun_single', _fun_52297)

        @norecursion
        def fun_vectorized(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fun_vectorized'
            module_type_store = module_type_store.open_function_context('fun_vectorized', 131, 12, False)
            
            # Passed parameters checking function
            fun_vectorized.stypy_localization = localization
            fun_vectorized.stypy_type_of_self = None
            fun_vectorized.stypy_type_store = module_type_store
            fun_vectorized.stypy_function_name = 'fun_vectorized'
            fun_vectorized.stypy_param_names_list = ['t', 'y']
            fun_vectorized.stypy_varargs_param_name = None
            fun_vectorized.stypy_kwargs_param_name = None
            fun_vectorized.stypy_call_defaults = defaults
            fun_vectorized.stypy_call_varargs = varargs
            fun_vectorized.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fun_vectorized', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fun_vectorized', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fun_vectorized(...)' code ##################

            
            # Assigning a Call to a Name (line 132):
            
            # Assigning a Call to a Name (line 132):
            
            # Call to empty_like(...): (line 132)
            # Processing the call arguments (line 132)
            # Getting the type of 'y' (line 132)
            y_52300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'y', False)
            # Processing the call keyword arguments (line 132)
            kwargs_52301 = {}
            # Getting the type of 'np' (line 132)
            np_52298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'np', False)
            # Obtaining the member 'empty_like' of a type (line 132)
            empty_like_52299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 20), np_52298, 'empty_like')
            # Calling empty_like(args, kwargs) (line 132)
            empty_like_call_result_52302 = invoke(stypy.reporting.localization.Localization(__file__, 132, 20), empty_like_52299, *[y_52300], **kwargs_52301)
            
            # Assigning a type to the variable 'f' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'f', empty_like_call_result_52302)
            
            
            # Call to enumerate(...): (line 133)
            # Processing the call arguments (line 133)
            # Getting the type of 'y' (line 133)
            y_52304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 39), 'y', False)
            # Obtaining the member 'T' of a type (line 133)
            T_52305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 39), y_52304, 'T')
            # Processing the call keyword arguments (line 133)
            kwargs_52306 = {}
            # Getting the type of 'enumerate' (line 133)
            enumerate_52303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 133)
            enumerate_call_result_52307 = invoke(stypy.reporting.localization.Localization(__file__, 133, 29), enumerate_52303, *[T_52305], **kwargs_52306)
            
            # Testing the type of a for loop iterable (line 133)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 133, 16), enumerate_call_result_52307)
            # Getting the type of the for loop variable (line 133)
            for_loop_var_52308 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 133, 16), enumerate_call_result_52307)
            # Assigning a type to the variable 'i' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 16), for_loop_var_52308))
            # Assigning a type to the variable 'yi' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'yi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 16), for_loop_var_52308))
            # SSA begins for a for statement (line 133)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Subscript (line 134):
            
            # Assigning a Call to a Subscript (line 134):
            
            # Call to _fun(...): (line 134)
            # Processing the call arguments (line 134)
            # Getting the type of 't' (line 134)
            t_52311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 40), 't', False)
            # Getting the type of 'yi' (line 134)
            yi_52312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 43), 'yi', False)
            # Processing the call keyword arguments (line 134)
            kwargs_52313 = {}
            # Getting the type of 'self' (line 134)
            self_52309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 30), 'self', False)
            # Obtaining the member '_fun' of a type (line 134)
            _fun_52310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 30), self_52309, '_fun')
            # Calling _fun(args, kwargs) (line 134)
            _fun_call_result_52314 = invoke(stypy.reporting.localization.Localization(__file__, 134, 30), _fun_52310, *[t_52311, yi_52312], **kwargs_52313)
            
            # Getting the type of 'f' (line 134)
            f_52315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'f')
            slice_52316 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 20), None, None, None)
            # Getting the type of 'i' (line 134)
            i_52317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'i')
            # Storing an element on a container (line 134)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 20), f_52315, ((slice_52316, i_52317), _fun_call_result_52314))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'f' (line 135)
            f_52318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'f')
            # Assigning a type to the variable 'stypy_return_type' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'stypy_return_type', f_52318)
            
            # ################# End of 'fun_vectorized(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fun_vectorized' in the type store
            # Getting the type of 'stypy_return_type' (line 131)
            stypy_return_type_52319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_52319)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fun_vectorized'
            return stypy_return_type_52319

        # Assigning a type to the variable 'fun_vectorized' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'fun_vectorized', fun_vectorized)
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def fun(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fun'
            module_type_store = module_type_store.open_function_context('fun', 137, 8, False)
            
            # Passed parameters checking function
            fun.stypy_localization = localization
            fun.stypy_type_of_self = None
            fun.stypy_type_store = module_type_store
            fun.stypy_function_name = 'fun'
            fun.stypy_param_names_list = ['t', 'y']
            fun.stypy_varargs_param_name = None
            fun.stypy_kwargs_param_name = None
            fun.stypy_call_defaults = defaults
            fun.stypy_call_varargs = varargs
            fun.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fun', ['t', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fun', localization, ['t', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fun(...)' code ##################

            
            # Getting the type of 'self' (line 138)
            self_52320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'self')
            # Obtaining the member 'nfev' of a type (line 138)
            nfev_52321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), self_52320, 'nfev')
            int_52322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 25), 'int')
            # Applying the binary operator '+=' (line 138)
            result_iadd_52323 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 12), '+=', nfev_52321, int_52322)
            # Getting the type of 'self' (line 138)
            self_52324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'self')
            # Setting the type of the member 'nfev' of a type (line 138)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), self_52324, 'nfev', result_iadd_52323)
            
            
            # Call to fun_single(...): (line 139)
            # Processing the call arguments (line 139)
            # Getting the type of 't' (line 139)
            t_52327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 't', False)
            # Getting the type of 'y' (line 139)
            y_52328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'y', False)
            # Processing the call keyword arguments (line 139)
            kwargs_52329 = {}
            # Getting the type of 'self' (line 139)
            self_52325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'self', False)
            # Obtaining the member 'fun_single' of a type (line 139)
            fun_single_52326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 19), self_52325, 'fun_single')
            # Calling fun_single(args, kwargs) (line 139)
            fun_single_call_result_52330 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), fun_single_52326, *[t_52327, y_52328], **kwargs_52329)
            
            # Assigning a type to the variable 'stypy_return_type' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'stypy_return_type', fun_single_call_result_52330)
            
            # ################# End of 'fun(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fun' in the type store
            # Getting the type of 'stypy_return_type' (line 137)
            stypy_return_type_52331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_52331)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fun'
            return stypy_return_type_52331

        # Assigning a type to the variable 'fun' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'fun', fun)
        
        # Assigning a Name to a Attribute (line 141):
        
        # Assigning a Name to a Attribute (line 141):
        # Getting the type of 'fun' (line 141)
        fun_52332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'fun')
        # Getting the type of 'self' (line 141)
        self_52333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self')
        # Setting the type of the member 'fun' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_52333, 'fun', fun_52332)
        
        # Assigning a Name to a Attribute (line 142):
        
        # Assigning a Name to a Attribute (line 142):
        # Getting the type of 'fun_single' (line 142)
        fun_single_52334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'fun_single')
        # Getting the type of 'self' (line 142)
        self_52335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Setting the type of the member 'fun_single' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_52335, 'fun_single', fun_single_52334)
        
        # Assigning a Name to a Attribute (line 143):
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'fun_vectorized' (line 143)
        fun_vectorized_52336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'fun_vectorized')
        # Getting the type of 'self' (line 143)
        self_52337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member 'fun_vectorized' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_52337, 'fun_vectorized', fun_vectorized_52336)
        
        # Assigning a IfExp to a Attribute (line 145):
        
        # Assigning a IfExp to a Attribute (line 145):
        
        
        # Getting the type of 't_bound' (line 145)
        t_bound_52338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 50), 't_bound')
        # Getting the type of 't0' (line 145)
        t0_52339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 61), 't0')
        # Applying the binary operator '!=' (line 145)
        result_ne_52340 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 50), '!=', t_bound_52338, t0_52339)
        
        # Testing the type of an if expression (line 145)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 25), result_ne_52340)
        # SSA begins for if expression (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to sign(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 't_bound' (line 145)
        t_bound_52343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), 't_bound', False)
        # Getting the type of 't0' (line 145)
        t0_52344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 43), 't0', False)
        # Applying the binary operator '-' (line 145)
        result_sub_52345 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 33), '-', t_bound_52343, t0_52344)
        
        # Processing the call keyword arguments (line 145)
        kwargs_52346 = {}
        # Getting the type of 'np' (line 145)
        np_52341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'np', False)
        # Obtaining the member 'sign' of a type (line 145)
        sign_52342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), np_52341, 'sign')
        # Calling sign(args, kwargs) (line 145)
        sign_call_result_52347 = invoke(stypy.reporting.localization.Localization(__file__, 145, 25), sign_52342, *[result_sub_52345], **kwargs_52346)
        
        # SSA branch for the else part of an if expression (line 145)
        module_type_store.open_ssa_branch('if expression else')
        int_52348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 69), 'int')
        # SSA join for if expression (line 145)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_52349 = union_type.UnionType.add(sign_call_result_52347, int_52348)
        
        # Getting the type of 'self' (line 145)
        self_52350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'direction' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_52350, 'direction', if_exp_52349)
        
        # Assigning a Attribute to a Attribute (line 146):
        
        # Assigning a Attribute to a Attribute (line 146):
        # Getting the type of 'self' (line 146)
        self_52351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'self')
        # Obtaining the member 'y' of a type (line 146)
        y_52352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 17), self_52351, 'y')
        # Obtaining the member 'size' of a type (line 146)
        size_52353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 17), y_52352, 'size')
        # Getting the type of 'self' (line 146)
        self_52354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'n' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_52354, 'n', size_52353)
        
        # Assigning a Str to a Attribute (line 147):
        
        # Assigning a Str to a Attribute (line 147):
        str_52355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 22), 'str', 'running')
        # Getting the type of 'self' (line 147)
        self_52356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member 'status' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_52356, 'status', str_52355)
        
        # Assigning a Num to a Attribute (line 149):
        
        # Assigning a Num to a Attribute (line 149):
        int_52357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'int')
        # Getting the type of 'self' (line 149)
        self_52358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Setting the type of the member 'nfev' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_52358, 'nfev', int_52357)
        
        # Assigning a Num to a Attribute (line 150):
        
        # Assigning a Num to a Attribute (line 150):
        int_52359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'int')
        # Getting the type of 'self' (line 150)
        self_52360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'njev' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_52360, 'njev', int_52359)
        
        # Assigning a Num to a Attribute (line 151):
        
        # Assigning a Num to a Attribute (line 151):
        int_52361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 19), 'int')
        # Getting the type of 'self' (line 151)
        self_52362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'nlu' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_52362, 'nlu', int_52361)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def step_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'step_size'
        module_type_store = module_type_store.open_function_context('step_size', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OdeSolver.step_size.__dict__.__setitem__('stypy_localization', localization)
        OdeSolver.step_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OdeSolver.step_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        OdeSolver.step_size.__dict__.__setitem__('stypy_function_name', 'OdeSolver.step_size')
        OdeSolver.step_size.__dict__.__setitem__('stypy_param_names_list', [])
        OdeSolver.step_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        OdeSolver.step_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OdeSolver.step_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        OdeSolver.step_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        OdeSolver.step_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OdeSolver.step_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolver.step_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'step_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'step_size(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 155)
        # Getting the type of 'self' (line 155)
        self_52363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'self')
        # Obtaining the member 't_old' of a type (line 155)
        t_old_52364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 11), self_52363, 't_old')
        # Getting the type of 'None' (line 155)
        None_52365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), 'None')
        
        (may_be_52366, more_types_in_union_52367) = may_be_none(t_old_52364, None_52365)

        if may_be_52366:

            if more_types_in_union_52367:
                # Runtime conditional SSA (line 155)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 156)
            None_52368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'stypy_return_type', None_52368)

            if more_types_in_union_52367:
                # Runtime conditional SSA for else branch (line 155)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_52366) or more_types_in_union_52367):
            
            # Call to abs(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'self' (line 158)
            self_52371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'self', False)
            # Obtaining the member 't' of a type (line 158)
            t_52372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 26), self_52371, 't')
            # Getting the type of 'self' (line 158)
            self_52373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 35), 'self', False)
            # Obtaining the member 't_old' of a type (line 158)
            t_old_52374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 35), self_52373, 't_old')
            # Applying the binary operator '-' (line 158)
            result_sub_52375 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 26), '-', t_52372, t_old_52374)
            
            # Processing the call keyword arguments (line 158)
            kwargs_52376 = {}
            # Getting the type of 'np' (line 158)
            np_52369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'np', False)
            # Obtaining the member 'abs' of a type (line 158)
            abs_52370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 19), np_52369, 'abs')
            # Calling abs(args, kwargs) (line 158)
            abs_call_result_52377 = invoke(stypy.reporting.localization.Localization(__file__, 158, 19), abs_52370, *[result_sub_52375], **kwargs_52376)
            
            # Assigning a type to the variable 'stypy_return_type' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'stypy_return_type', abs_call_result_52377)

            if (may_be_52366 and more_types_in_union_52367):
                # SSA join for if statement (line 155)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'step_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'step_size' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_52378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52378)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'step_size'
        return stypy_return_type_52378


    @norecursion
    def step(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'step'
        module_type_store = module_type_store.open_function_context('step', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OdeSolver.step.__dict__.__setitem__('stypy_localization', localization)
        OdeSolver.step.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OdeSolver.step.__dict__.__setitem__('stypy_type_store', module_type_store)
        OdeSolver.step.__dict__.__setitem__('stypy_function_name', 'OdeSolver.step')
        OdeSolver.step.__dict__.__setitem__('stypy_param_names_list', [])
        OdeSolver.step.__dict__.__setitem__('stypy_varargs_param_name', None)
        OdeSolver.step.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OdeSolver.step.__dict__.__setitem__('stypy_call_defaults', defaults)
        OdeSolver.step.__dict__.__setitem__('stypy_call_varargs', varargs)
        OdeSolver.step.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OdeSolver.step.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolver.step', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'step', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'step(...)' code ##################

        str_52379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, (-1)), 'str', "Perform one integration step.\n\n        Returns\n        -------\n        message : string or None\n            Report from the solver. Typically a reason for a failure if\n            `self.status` is 'failed' after the step was taken or None\n            otherwise.\n        ")
        
        
        # Getting the type of 'self' (line 170)
        self_52380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'self')
        # Obtaining the member 'status' of a type (line 170)
        status_52381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), self_52380, 'status')
        str_52382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 26), 'str', 'running')
        # Applying the binary operator '!=' (line 170)
        result_ne_52383 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 11), '!=', status_52381, str_52382)
        
        # Testing the type of an if condition (line 170)
        if_condition_52384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 8), result_ne_52383)
        # Assigning a type to the variable 'if_condition_52384' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'if_condition_52384', if_condition_52384)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 171)
        # Processing the call arguments (line 171)
        str_52386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 31), 'str', 'Attempt to step on a failed or finished solver.')
        # Processing the call keyword arguments (line 171)
        kwargs_52387 = {}
        # Getting the type of 'RuntimeError' (line 171)
        RuntimeError_52385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 171)
        RuntimeError_call_result_52388 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), RuntimeError_52385, *[str_52386], **kwargs_52387)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 171, 12), RuntimeError_call_result_52388, 'raise parameter', BaseException)
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 174)
        self_52389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'self')
        # Obtaining the member 'n' of a type (line 174)
        n_52390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 11), self_52389, 'n')
        int_52391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 21), 'int')
        # Applying the binary operator '==' (line 174)
        result_eq_52392 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 11), '==', n_52390, int_52391)
        
        
        # Getting the type of 'self' (line 174)
        self_52393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'self')
        # Obtaining the member 't' of a type (line 174)
        t_52394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), self_52393, 't')
        # Getting the type of 'self' (line 174)
        self_52395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 36), 'self')
        # Obtaining the member 't_bound' of a type (line 174)
        t_bound_52396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 36), self_52395, 't_bound')
        # Applying the binary operator '==' (line 174)
        result_eq_52397 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 26), '==', t_52394, t_bound_52396)
        
        # Applying the binary operator 'or' (line 174)
        result_or_keyword_52398 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 11), 'or', result_eq_52392, result_eq_52397)
        
        # Testing the type of an if condition (line 174)
        if_condition_52399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 8), result_or_keyword_52398)
        # Assigning a type to the variable 'if_condition_52399' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'if_condition_52399', if_condition_52399)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 176):
        
        # Assigning a Attribute to a Attribute (line 176):
        # Getting the type of 'self' (line 176)
        self_52400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'self')
        # Obtaining the member 't' of a type (line 176)
        t_52401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), self_52400, 't')
        # Getting the type of 'self' (line 176)
        self_52402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self')
        # Setting the type of the member 't_old' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_52402, 't_old', t_52401)
        
        # Assigning a Attribute to a Attribute (line 177):
        
        # Assigning a Attribute to a Attribute (line 177):
        # Getting the type of 'self' (line 177)
        self_52403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'self')
        # Obtaining the member 't_bound' of a type (line 177)
        t_bound_52404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 21), self_52403, 't_bound')
        # Getting the type of 'self' (line 177)
        self_52405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'self')
        # Setting the type of the member 't' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), self_52405, 't', t_bound_52404)
        
        # Assigning a Name to a Name (line 178):
        
        # Assigning a Name to a Name (line 178):
        # Getting the type of 'None' (line 178)
        None_52406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'None')
        # Assigning a type to the variable 'message' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'message', None_52406)
        
        # Assigning a Str to a Attribute (line 179):
        
        # Assigning a Str to a Attribute (line 179):
        str_52407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 26), 'str', 'finished')
        # Getting the type of 'self' (line 179)
        self_52408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'self')
        # Setting the type of the member 'status' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), self_52408, 'status', str_52407)
        # SSA branch for the else part of an if statement (line 174)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 181):
        
        # Assigning a Attribute to a Name (line 181):
        # Getting the type of 'self' (line 181)
        self_52409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'self')
        # Obtaining the member 't' of a type (line 181)
        t_52410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), self_52409, 't')
        # Assigning a type to the variable 't' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 't', t_52410)
        
        # Assigning a Call to a Tuple (line 182):
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_52411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 12), 'int')
        
        # Call to _step_impl(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_52414 = {}
        # Getting the type of 'self' (line 182)
        self_52412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'self', False)
        # Obtaining the member '_step_impl' of a type (line 182)
        _step_impl_52413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 31), self_52412, '_step_impl')
        # Calling _step_impl(args, kwargs) (line 182)
        _step_impl_call_result_52415 = invoke(stypy.reporting.localization.Localization(__file__, 182, 31), _step_impl_52413, *[], **kwargs_52414)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___52416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), _step_impl_call_result_52415, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_52417 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), getitem___52416, int_52411)
        
        # Assigning a type to the variable 'tuple_var_assignment_52186' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'tuple_var_assignment_52186', subscript_call_result_52417)
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_52418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 12), 'int')
        
        # Call to _step_impl(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_52421 = {}
        # Getting the type of 'self' (line 182)
        self_52419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'self', False)
        # Obtaining the member '_step_impl' of a type (line 182)
        _step_impl_52420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 31), self_52419, '_step_impl')
        # Calling _step_impl(args, kwargs) (line 182)
        _step_impl_call_result_52422 = invoke(stypy.reporting.localization.Localization(__file__, 182, 31), _step_impl_52420, *[], **kwargs_52421)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___52423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), _step_impl_call_result_52422, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_52424 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), getitem___52423, int_52418)
        
        # Assigning a type to the variable 'tuple_var_assignment_52187' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'tuple_var_assignment_52187', subscript_call_result_52424)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_var_assignment_52186' (line 182)
        tuple_var_assignment_52186_52425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'tuple_var_assignment_52186')
        # Assigning a type to the variable 'success' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'success', tuple_var_assignment_52186_52425)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_var_assignment_52187' (line 182)
        tuple_var_assignment_52187_52426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'tuple_var_assignment_52187')
        # Assigning a type to the variable 'message' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 21), 'message', tuple_var_assignment_52187_52426)
        
        
        # Getting the type of 'success' (line 184)
        success_52427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'success')
        # Applying the 'not' unary operator (line 184)
        result_not__52428 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), 'not', success_52427)
        
        # Testing the type of an if condition (line 184)
        if_condition_52429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 12), result_not__52428)
        # Assigning a type to the variable 'if_condition_52429' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'if_condition_52429', if_condition_52429)
        # SSA begins for if statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 185):
        
        # Assigning a Str to a Attribute (line 185):
        str_52430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'str', 'failed')
        # Getting the type of 'self' (line 185)
        self_52431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'self')
        # Setting the type of the member 'status' of a type (line 185)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 16), self_52431, 'status', str_52430)
        # SSA branch for the else part of an if statement (line 184)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 187):
        
        # Assigning a Name to a Attribute (line 187):
        # Getting the type of 't' (line 187)
        t_52432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 29), 't')
        # Getting the type of 'self' (line 187)
        self_52433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'self')
        # Setting the type of the member 't_old' of a type (line 187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), self_52433, 't_old', t_52432)
        
        
        # Getting the type of 'self' (line 188)
        self_52434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'self')
        # Obtaining the member 'direction' of a type (line 188)
        direction_52435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 19), self_52434, 'direction')
        # Getting the type of 'self' (line 188)
        self_52436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 37), 'self')
        # Obtaining the member 't' of a type (line 188)
        t_52437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 37), self_52436, 't')
        # Getting the type of 'self' (line 188)
        self_52438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 46), 'self')
        # Obtaining the member 't_bound' of a type (line 188)
        t_bound_52439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 46), self_52438, 't_bound')
        # Applying the binary operator '-' (line 188)
        result_sub_52440 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 37), '-', t_52437, t_bound_52439)
        
        # Applying the binary operator '*' (line 188)
        result_mul_52441 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 19), '*', direction_52435, result_sub_52440)
        
        int_52442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 63), 'int')
        # Applying the binary operator '>=' (line 188)
        result_ge_52443 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 19), '>=', result_mul_52441, int_52442)
        
        # Testing the type of an if condition (line 188)
        if_condition_52444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 16), result_ge_52443)
        # Assigning a type to the variable 'if_condition_52444' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'if_condition_52444', if_condition_52444)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 189):
        
        # Assigning a Str to a Attribute (line 189):
        str_52445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 34), 'str', 'finished')
        # Getting the type of 'self' (line 189)
        self_52446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'self')
        # Setting the type of the member 'status' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 20), self_52446, 'status', str_52445)
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'message' (line 191)
        message_52447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'message')
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', message_52447)
        
        # ################# End of 'step(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'step' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_52448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52448)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'step'
        return stypy_return_type_52448


    @norecursion
    def dense_output(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dense_output'
        module_type_store = module_type_store.open_function_context('dense_output', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OdeSolver.dense_output.__dict__.__setitem__('stypy_localization', localization)
        OdeSolver.dense_output.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OdeSolver.dense_output.__dict__.__setitem__('stypy_type_store', module_type_store)
        OdeSolver.dense_output.__dict__.__setitem__('stypy_function_name', 'OdeSolver.dense_output')
        OdeSolver.dense_output.__dict__.__setitem__('stypy_param_names_list', [])
        OdeSolver.dense_output.__dict__.__setitem__('stypy_varargs_param_name', None)
        OdeSolver.dense_output.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OdeSolver.dense_output.__dict__.__setitem__('stypy_call_defaults', defaults)
        OdeSolver.dense_output.__dict__.__setitem__('stypy_call_varargs', varargs)
        OdeSolver.dense_output.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OdeSolver.dense_output.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolver.dense_output', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dense_output', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dense_output(...)' code ##################

        str_52449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, (-1)), 'str', 'Compute a local interpolant over the last successful step.\n\n        Returns\n        -------\n        sol : `DenseOutput`\n            Local interpolant over the last successful step.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 201)
        # Getting the type of 'self' (line 201)
        self_52450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'self')
        # Obtaining the member 't_old' of a type (line 201)
        t_old_52451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), self_52450, 't_old')
        # Getting the type of 'None' (line 201)
        None_52452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'None')
        
        (may_be_52453, more_types_in_union_52454) = may_be_none(t_old_52451, None_52452)

        if may_be_52453:

            if more_types_in_union_52454:
                # Runtime conditional SSA (line 201)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to RuntimeError(...): (line 202)
            # Processing the call arguments (line 202)
            str_52456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 31), 'str', 'Dense output is available after a successful step was made.')
            # Processing the call keyword arguments (line 202)
            kwargs_52457 = {}
            # Getting the type of 'RuntimeError' (line 202)
            RuntimeError_52455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 202)
            RuntimeError_call_result_52458 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), RuntimeError_52455, *[str_52456], **kwargs_52457)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 202, 12), RuntimeError_call_result_52458, 'raise parameter', BaseException)

            if more_types_in_union_52454:
                # SSA join for if statement (line 201)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 205)
        self_52459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'self')
        # Obtaining the member 'n' of a type (line 205)
        n_52460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 11), self_52459, 'n')
        int_52461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 21), 'int')
        # Applying the binary operator '==' (line 205)
        result_eq_52462 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), '==', n_52460, int_52461)
        
        
        # Getting the type of 'self' (line 205)
        self_52463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'self')
        # Obtaining the member 't' of a type (line 205)
        t_52464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 26), self_52463, 't')
        # Getting the type of 'self' (line 205)
        self_52465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 36), 'self')
        # Obtaining the member 't_old' of a type (line 205)
        t_old_52466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 36), self_52465, 't_old')
        # Applying the binary operator '==' (line 205)
        result_eq_52467 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 26), '==', t_52464, t_old_52466)
        
        # Applying the binary operator 'or' (line 205)
        result_or_keyword_52468 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), 'or', result_eq_52462, result_eq_52467)
        
        # Testing the type of an if condition (line 205)
        if_condition_52469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_or_keyword_52468)
        # Assigning a type to the variable 'if_condition_52469' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_52469', if_condition_52469)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ConstantDenseOutput(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'self' (line 207)
        self_52471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 39), 'self', False)
        # Obtaining the member 't_old' of a type (line 207)
        t_old_52472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 39), self_52471, 't_old')
        # Getting the type of 'self' (line 207)
        self_52473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 51), 'self', False)
        # Obtaining the member 't' of a type (line 207)
        t_52474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 51), self_52473, 't')
        # Getting the type of 'self' (line 207)
        self_52475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 59), 'self', False)
        # Obtaining the member 'y' of a type (line 207)
        y_52476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 59), self_52475, 'y')
        # Processing the call keyword arguments (line 207)
        kwargs_52477 = {}
        # Getting the type of 'ConstantDenseOutput' (line 207)
        ConstantDenseOutput_52470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'ConstantDenseOutput', False)
        # Calling ConstantDenseOutput(args, kwargs) (line 207)
        ConstantDenseOutput_call_result_52478 = invoke(stypy.reporting.localization.Localization(__file__, 207, 19), ConstantDenseOutput_52470, *[t_old_52472, t_52474, y_52476], **kwargs_52477)
        
        # Assigning a type to the variable 'stypy_return_type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'stypy_return_type', ConstantDenseOutput_call_result_52478)
        # SSA branch for the else part of an if statement (line 205)
        module_type_store.open_ssa_branch('else')
        
        # Call to _dense_output_impl(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_52481 = {}
        # Getting the type of 'self' (line 209)
        self_52479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'self', False)
        # Obtaining the member '_dense_output_impl' of a type (line 209)
        _dense_output_impl_52480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 19), self_52479, '_dense_output_impl')
        # Calling _dense_output_impl(args, kwargs) (line 209)
        _dense_output_impl_call_result_52482 = invoke(stypy.reporting.localization.Localization(__file__, 209, 19), _dense_output_impl_52480, *[], **kwargs_52481)
        
        # Assigning a type to the variable 'stypy_return_type' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'stypy_return_type', _dense_output_impl_call_result_52482)
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'dense_output(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dense_output' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_52483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52483)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dense_output'
        return stypy_return_type_52483


    @norecursion
    def _step_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_step_impl'
        module_type_store = module_type_store.open_function_context('_step_impl', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OdeSolver._step_impl.__dict__.__setitem__('stypy_localization', localization)
        OdeSolver._step_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OdeSolver._step_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        OdeSolver._step_impl.__dict__.__setitem__('stypy_function_name', 'OdeSolver._step_impl')
        OdeSolver._step_impl.__dict__.__setitem__('stypy_param_names_list', [])
        OdeSolver._step_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        OdeSolver._step_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OdeSolver._step_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        OdeSolver._step_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        OdeSolver._step_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OdeSolver._step_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolver._step_impl', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_step_impl', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_step_impl(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 212)
        NotImplementedError_52484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 212, 8), NotImplementedError_52484, 'raise parameter', BaseException)
        
        # ################# End of '_step_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_step_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_52485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52485)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_step_impl'
        return stypy_return_type_52485


    @norecursion
    def _dense_output_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dense_output_impl'
        module_type_store = module_type_store.open_function_context('_dense_output_impl', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_localization', localization)
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_function_name', 'OdeSolver._dense_output_impl')
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_param_names_list', [])
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OdeSolver._dense_output_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolver._dense_output_impl', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dense_output_impl', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dense_output_impl(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 215)
        NotImplementedError_52486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 215, 8), NotImplementedError_52486, 'raise parameter', BaseException)
        
        # ################# End of '_dense_output_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dense_output_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_52487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52487)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dense_output_impl'
        return stypy_return_type_52487


# Assigning a type to the variable 'OdeSolver' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'OdeSolver', OdeSolver)

# Assigning a Str to a Name (line 114):
str_52488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 21), 'str', 'Required step size is less than spacing between numbers.')
# Getting the type of 'OdeSolver'
OdeSolver_52489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'OdeSolver')
# Setting the type of the member 'TOO_SMALL_STEP' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), OdeSolver_52489, 'TOO_SMALL_STEP', str_52488)
# Declaration of the 'DenseOutput' class

class DenseOutput(object, ):
    str_52490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, (-1)), 'str', 'Base class for local interpolant over step made by an ODE solver.\n\n    It interpolates between `t_min` and `t_max` (see Attributes below).\n    Evaluation outside this interval is not forbidden, but the accuracy is not\n    guaranteed.\n\n    Attributes\n    ----------\n    t_min, t_max : float\n        Time range of the interpolation.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DenseOutput.__init__', ['t_old', 't'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['t_old', 't'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 231):
        
        # Assigning a Name to a Attribute (line 231):
        # Getting the type of 't_old' (line 231)
        t_old_52491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 't_old')
        # Getting the type of 'self' (line 231)
        self_52492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'self')
        # Setting the type of the member 't_old' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), self_52492, 't_old', t_old_52491)
        
        # Assigning a Name to a Attribute (line 232):
        
        # Assigning a Name to a Attribute (line 232):
        # Getting the type of 't' (line 232)
        t_52493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 't')
        # Getting the type of 'self' (line 232)
        self_52494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self')
        # Setting the type of the member 't' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_52494, 't', t_52493)
        
        # Assigning a Call to a Attribute (line 233):
        
        # Assigning a Call to a Attribute (line 233):
        
        # Call to min(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 't' (line 233)
        t_52496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 25), 't', False)
        # Getting the type of 't_old' (line 233)
        t_old_52497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 't_old', False)
        # Processing the call keyword arguments (line 233)
        kwargs_52498 = {}
        # Getting the type of 'min' (line 233)
        min_52495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'min', False)
        # Calling min(args, kwargs) (line 233)
        min_call_result_52499 = invoke(stypy.reporting.localization.Localization(__file__, 233, 21), min_52495, *[t_52496, t_old_52497], **kwargs_52498)
        
        # Getting the type of 'self' (line 233)
        self_52500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'self')
        # Setting the type of the member 't_min' of a type (line 233)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), self_52500, 't_min', min_call_result_52499)
        
        # Assigning a Call to a Attribute (line 234):
        
        # Assigning a Call to a Attribute (line 234):
        
        # Call to max(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 't' (line 234)
        t_52502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 't', False)
        # Getting the type of 't_old' (line 234)
        t_old_52503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 28), 't_old', False)
        # Processing the call keyword arguments (line 234)
        kwargs_52504 = {}
        # Getting the type of 'max' (line 234)
        max_52501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'max', False)
        # Calling max(args, kwargs) (line 234)
        max_call_result_52505 = invoke(stypy.reporting.localization.Localization(__file__, 234, 21), max_52501, *[t_52502, t_old_52503], **kwargs_52504)
        
        # Getting the type of 'self' (line 234)
        self_52506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self')
        # Setting the type of the member 't_max' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_52506, 't_max', max_call_result_52505)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DenseOutput.__call__.__dict__.__setitem__('stypy_localization', localization)
        DenseOutput.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DenseOutput.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DenseOutput.__call__.__dict__.__setitem__('stypy_function_name', 'DenseOutput.__call__')
        DenseOutput.__call__.__dict__.__setitem__('stypy_param_names_list', ['t'])
        DenseOutput.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DenseOutput.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DenseOutput.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DenseOutput.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DenseOutput.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DenseOutput.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DenseOutput.__call__', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_52507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, (-1)), 'str', 'Evaluate the interpolant.\n\n        Parameters\n        ----------\n        t : float or array_like with shape (n_points,)\n            Points to evaluate the solution at.\n\n        Returns\n        -------\n        y : ndarray, shape (n,) or (n, n_points)\n            Computed values. Shape depends on whether `t` was a scalar or a\n            1-d array.\n        ')
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to asarray(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 't' (line 250)
        t_52510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 23), 't', False)
        # Processing the call keyword arguments (line 250)
        kwargs_52511 = {}
        # Getting the type of 'np' (line 250)
        np_52508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 250)
        asarray_52509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), np_52508, 'asarray')
        # Calling asarray(args, kwargs) (line 250)
        asarray_call_result_52512 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), asarray_52509, *[t_52510], **kwargs_52511)
        
        # Assigning a type to the variable 't' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 't', asarray_call_result_52512)
        
        
        # Getting the type of 't' (line 251)
        t_52513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 't')
        # Obtaining the member 'ndim' of a type (line 251)
        ndim_52514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 11), t_52513, 'ndim')
        int_52515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'int')
        # Applying the binary operator '>' (line 251)
        result_gt_52516 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 11), '>', ndim_52514, int_52515)
        
        # Testing the type of an if condition (line 251)
        if_condition_52517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), result_gt_52516)
        # Assigning a type to the variable 'if_condition_52517' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_52517', if_condition_52517)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 252)
        # Processing the call arguments (line 252)
        str_52519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 29), 'str', '`t` must be float or 1-d array.')
        # Processing the call keyword arguments (line 252)
        kwargs_52520 = {}
        # Getting the type of 'ValueError' (line 252)
        ValueError_52518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 252)
        ValueError_call_result_52521 = invoke(stypy.reporting.localization.Localization(__file__, 252, 18), ValueError_52518, *[str_52519], **kwargs_52520)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 252, 12), ValueError_call_result_52521, 'raise parameter', BaseException)
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _call_impl(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 't' (line 253)
        t_52524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 31), 't', False)
        # Processing the call keyword arguments (line 253)
        kwargs_52525 = {}
        # Getting the type of 'self' (line 253)
        self_52522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'self', False)
        # Obtaining the member '_call_impl' of a type (line 253)
        _call_impl_52523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 15), self_52522, '_call_impl')
        # Calling _call_impl(args, kwargs) (line 253)
        _call_impl_call_result_52526 = invoke(stypy.reporting.localization.Localization(__file__, 253, 15), _call_impl_52523, *[t_52524], **kwargs_52525)
        
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'stypy_return_type', _call_impl_call_result_52526)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_52527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52527)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_52527


    @norecursion
    def _call_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_impl'
        module_type_store = module_type_store.open_function_context('_call_impl', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DenseOutput._call_impl.__dict__.__setitem__('stypy_localization', localization)
        DenseOutput._call_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DenseOutput._call_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        DenseOutput._call_impl.__dict__.__setitem__('stypy_function_name', 'DenseOutput._call_impl')
        DenseOutput._call_impl.__dict__.__setitem__('stypy_param_names_list', ['t'])
        DenseOutput._call_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        DenseOutput._call_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DenseOutput._call_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        DenseOutput._call_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        DenseOutput._call_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DenseOutput._call_impl.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DenseOutput._call_impl', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_impl', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_impl(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 256)
        NotImplementedError_52528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 8), NotImplementedError_52528, 'raise parameter', BaseException)
        
        # ################# End of '_call_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_52529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52529)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_impl'
        return stypy_return_type_52529


# Assigning a type to the variable 'DenseOutput' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'DenseOutput', DenseOutput)
# Declaration of the 'ConstantDenseOutput' class
# Getting the type of 'DenseOutput' (line 259)
DenseOutput_52530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'DenseOutput')

class ConstantDenseOutput(DenseOutput_52530, ):
    str_52531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, (-1)), 'str', 'Constant value interpolator.\n\n    This class used for degenerate integration cases: equal integration limits\n    or a system with 0 equations.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 265, 4, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConstantDenseOutput.__init__', ['t_old', 't', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['t_old', 't', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 't_old' (line 266)
        t_old_52538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 50), 't_old', False)
        # Getting the type of 't' (line 266)
        t_52539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 57), 't', False)
        # Processing the call keyword arguments (line 266)
        kwargs_52540 = {}
        
        # Call to super(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'ConstantDenseOutput' (line 266)
        ConstantDenseOutput_52533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 14), 'ConstantDenseOutput', False)
        # Getting the type of 'self' (line 266)
        self_52534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 35), 'self', False)
        # Processing the call keyword arguments (line 266)
        kwargs_52535 = {}
        # Getting the type of 'super' (line 266)
        super_52532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'super', False)
        # Calling super(args, kwargs) (line 266)
        super_call_result_52536 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), super_52532, *[ConstantDenseOutput_52533, self_52534], **kwargs_52535)
        
        # Obtaining the member '__init__' of a type (line 266)
        init___52537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), super_call_result_52536, '__init__')
        # Calling __init__(args, kwargs) (line 266)
        init___call_result_52541 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), init___52537, *[t_old_52538, t_52539], **kwargs_52540)
        
        
        # Assigning a Name to a Attribute (line 267):
        
        # Assigning a Name to a Attribute (line 267):
        # Getting the type of 'value' (line 267)
        value_52542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'value')
        # Getting the type of 'self' (line 267)
        self_52543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self')
        # Setting the type of the member 'value' of a type (line 267)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_52543, 'value', value_52542)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _call_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_impl'
        module_type_store = module_type_store.open_function_context('_call_impl', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_localization', localization)
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_function_name', 'ConstantDenseOutput._call_impl')
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_param_names_list', ['t'])
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConstantDenseOutput._call_impl.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConstantDenseOutput._call_impl', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_impl', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_impl(...)' code ##################

        
        
        # Getting the type of 't' (line 270)
        t_52544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 't')
        # Obtaining the member 'ndim' of a type (line 270)
        ndim_52545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 11), t_52544, 'ndim')
        int_52546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 21), 'int')
        # Applying the binary operator '==' (line 270)
        result_eq_52547 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 11), '==', ndim_52545, int_52546)
        
        # Testing the type of an if condition (line 270)
        if_condition_52548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 8), result_eq_52547)
        # Assigning a type to the variable 'if_condition_52548' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'if_condition_52548', if_condition_52548)
        # SSA begins for if statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 271)
        self_52549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'self')
        # Obtaining the member 'value' of a type (line 271)
        value_52550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 19), self_52549, 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'stypy_return_type', value_52550)
        # SSA branch for the else part of an if statement (line 270)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to empty(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Obtaining an instance of the builtin type 'tuple' (line 273)
        tuple_52553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 273)
        # Adding element type (line 273)
        
        # Obtaining the type of the subscript
        int_52554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 45), 'int')
        # Getting the type of 'self' (line 273)
        self_52555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), 'self', False)
        # Obtaining the member 'value' of a type (line 273)
        value_52556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 28), self_52555, 'value')
        # Obtaining the member 'shape' of a type (line 273)
        shape_52557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 28), value_52556, 'shape')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___52558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 28), shape_52557, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_52559 = invoke(stypy.reporting.localization.Localization(__file__, 273, 28), getitem___52558, int_52554)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 28), tuple_52553, subscript_call_result_52559)
        # Adding element type (line 273)
        
        # Obtaining the type of the subscript
        int_52560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 57), 'int')
        # Getting the type of 't' (line 273)
        t_52561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 49), 't', False)
        # Obtaining the member 'shape' of a type (line 273)
        shape_52562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 49), t_52561, 'shape')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___52563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 49), shape_52562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_52564 = invoke(stypy.reporting.localization.Localization(__file__, 273, 49), getitem___52563, int_52560)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 28), tuple_52553, subscript_call_result_52564)
        
        # Processing the call keyword arguments (line 273)
        kwargs_52565 = {}
        # Getting the type of 'np' (line 273)
        np_52551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 18), 'np', False)
        # Obtaining the member 'empty' of a type (line 273)
        empty_52552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 18), np_52551, 'empty')
        # Calling empty(args, kwargs) (line 273)
        empty_call_result_52566 = invoke(stypy.reporting.localization.Localization(__file__, 273, 18), empty_52552, *[tuple_52553], **kwargs_52565)
        
        # Assigning a type to the variable 'ret' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'ret', empty_call_result_52566)
        
        # Assigning a Subscript to a Subscript (line 274):
        
        # Assigning a Subscript to a Subscript (line 274):
        
        # Obtaining the type of the subscript
        slice_52567 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 21), None, None, None)
        # Getting the type of 'None' (line 274)
        None_52568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'None')
        # Getting the type of 'self' (line 274)
        self_52569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 21), 'self')
        # Obtaining the member 'value' of a type (line 274)
        value_52570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 21), self_52569, 'value')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___52571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 21), value_52570, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_52572 = invoke(stypy.reporting.localization.Localization(__file__, 274, 21), getitem___52571, (slice_52567, None_52568))
        
        # Getting the type of 'ret' (line 274)
        ret_52573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'ret')
        slice_52574 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 12), None, None, None)
        # Storing an element on a container (line 274)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), ret_52573, (slice_52574, subscript_call_result_52572))
        # Getting the type of 'ret' (line 275)
        ret_52575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'stypy_return_type', ret_52575)
        # SSA join for if statement (line 270)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_call_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_52576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_impl'
        return stypy_return_type_52576


# Assigning a type to the variable 'ConstantDenseOutput' (line 259)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'ConstantDenseOutput', ConstantDenseOutput)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
