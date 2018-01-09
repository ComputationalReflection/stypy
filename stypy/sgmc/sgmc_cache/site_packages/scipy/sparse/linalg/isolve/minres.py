
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy import sqrt, inner, finfo, zeros
4: from numpy.linalg import norm
5: 
6: from .utils import make_system
7: from .iterative import set_docstring
8: 
9: __all__ = ['minres']
10: 
11: 
12: header = \
13: '''Use MINimum RESidual iteration to solve Ax=b
14: 
15: MINRES minimizes norm(A*x - b) for a real symmetric matrix A.  Unlike
16: the Conjugate Gradient method, A can be indefinite or singular.
17: 
18: If shift != 0 then the method solves (A - shift*I)x = b
19: '''
20: 
21: Ainfo = "The real symmetric N-by-N matrix of the linear system"
22: 
23: footer = \
24: '''
25: Notes
26: -----
27: THIS FUNCTION IS EXPERIMENTAL AND SUBJECT TO CHANGE!
28: 
29: References
30: ----------
31: Solution of sparse indefinite systems of linear equations,
32:     C. C. Paige and M. A. Saunders (1975),
33:     SIAM J. Numer. Anal. 12(4), pp. 617-629.
34:     https://web.stanford.edu/group/SOL/software/minres/
35: 
36: This file is a translation of the following MATLAB implementation:
37:     https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip
38: '''
39: 
40: 
41: @set_docstring(header,
42:                Ainfo,
43:                footer)
44: def minres(A, b, x0=None, shift=0.0, tol=1e-5, maxiter=None,
45:            M=None, callback=None, show=False, check=False):
46:     A, M, x, b, postprocess = make_system(A, M, x0, b)
47: 
48:     matvec = A.matvec
49:     psolve = M.matvec
50: 
51:     first = 'Enter minres.   '
52:     last = 'Exit  minres.   '
53: 
54:     n = A.shape[0]
55: 
56:     if maxiter is None:
57:         maxiter = 5 * n
58: 
59:     msg = [' beta2 = 0.  If M = I, b and x are eigenvectors    ',   # -1
60:             ' beta1 = 0.  The exact solution is  x = 0          ',   # 0
61:             ' A solution to Ax = b was found, given rtol        ',   # 1
62:             ' A least-squares solution was found, given rtol    ',   # 2
63:             ' Reasonable accuracy achieved, given eps           ',   # 3
64:             ' x has converged to an eigenvector                 ',   # 4
65:             ' acond has exceeded 0.1/eps                        ',   # 5
66:             ' The iteration limit was reached                   ',   # 6
67:             ' A  does not define a symmetric matrix             ',   # 7
68:             ' M  does not define a symmetric matrix             ',   # 8
69:             ' M  does not define a pos-def preconditioner       ']   # 9
70: 
71:     if show:
72:         print(first + 'Solution of symmetric Ax = b')
73:         print(first + 'n      =  %3g     shift  =  %23.14e' % (n,shift))
74:         print(first + 'itnlim =  %3g     rtol   =  %11.2e' % (maxiter,tol))
75:         print()
76: 
77:     istop = 0
78:     itn = 0
79:     Anorm = 0
80:     Acond = 0
81:     rnorm = 0
82:     ynorm = 0
83: 
84:     xtype = x.dtype
85: 
86:     eps = finfo(xtype).eps
87: 
88:     x = zeros(n, dtype=xtype)
89: 
90:     # Set up y and v for the first Lanczos vector v1.
91:     # y  =  beta1 P' v1,  where  P = C**(-1).
92:     # v is really P' v1.
93: 
94:     y = b
95:     r1 = b
96: 
97:     y = psolve(b)
98: 
99:     beta1 = inner(b,y)
100: 
101:     if beta1 < 0:
102:         raise ValueError('indefinite preconditioner')
103:     elif beta1 == 0:
104:         return (postprocess(x), 0)
105: 
106:     beta1 = sqrt(beta1)
107: 
108:     if check:
109:         # are these too strict?
110: 
111:         # see if A is symmetric
112:         w = matvec(y)
113:         r2 = matvec(w)
114:         s = inner(w,w)
115:         t = inner(y,r2)
116:         z = abs(s - t)
117:         epsa = (s + eps) * eps**(1.0/3.0)
118:         if z > epsa:
119:             raise ValueError('non-symmetric matrix')
120: 
121:         # see if M is symmetric
122:         r2 = psolve(y)
123:         s = inner(y,y)
124:         t = inner(r1,r2)
125:         z = abs(s - t)
126:         epsa = (s + eps) * eps**(1.0/3.0)
127:         if z > epsa:
128:             raise ValueError('non-symmetric preconditioner')
129: 
130:     # Initialize other quantities
131:     oldb = 0
132:     beta = beta1
133:     dbar = 0
134:     epsln = 0
135:     qrnorm = beta1
136:     phibar = beta1
137:     rhs1 = beta1
138:     rhs2 = 0
139:     tnorm2 = 0
140:     ynorm2 = 0
141:     cs = -1
142:     sn = 0
143:     w = zeros(n, dtype=xtype)
144:     w2 = zeros(n, dtype=xtype)
145:     r2 = r1
146: 
147:     if show:
148:         print()
149:         print()
150:         print('   Itn     x(1)     Compatible    LS       norm(A)  cond(A) gbar/|A|')
151: 
152:     while itn < maxiter:
153:         itn += 1
154: 
155:         s = 1.0/beta
156:         v = s*y
157: 
158:         y = matvec(v)
159:         y = y - shift * v
160: 
161:         if itn >= 2:
162:             y = y - (beta/oldb)*r1
163: 
164:         alfa = inner(v,y)
165:         y = y - (alfa/beta)*r2
166:         r1 = r2
167:         r2 = y
168:         y = psolve(r2)
169:         oldb = beta
170:         beta = inner(r2,y)
171:         if beta < 0:
172:             raise ValueError('non-symmetric matrix')
173:         beta = sqrt(beta)
174:         tnorm2 += alfa**2 + oldb**2 + beta**2
175: 
176:         if itn == 1:
177:             if beta/beta1 <= 10*eps:
178:                 istop = -1  # Terminate later
179:             # tnorm2 = alfa**2 ??
180:             gmax = abs(alfa)
181:             gmin = gmax
182: 
183:         # Apply previous rotation Qk-1 to get
184:         #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
185:         #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
186: 
187:         oldeps = epsln
188:         delta = cs * dbar + sn * alfa   # delta1 = 0         deltak
189:         gbar = sn * dbar - cs * alfa   # gbar 1 = alfa1     gbar k
190:         epsln = sn * beta     # epsln2 = 0         epslnk+1
191:         dbar = - cs * beta   # dbar 2 = beta2     dbar k+1
192:         root = norm([gbar, dbar])
193:         Arnorm = phibar * root
194: 
195:         # Compute the next plane rotation Qk
196: 
197:         gamma = norm([gbar, beta])       # gammak
198:         gamma = max(gamma, eps)
199:         cs = gbar / gamma             # ck
200:         sn = beta / gamma             # sk
201:         phi = cs * phibar              # phik
202:         phibar = sn * phibar              # phibark+1
203: 
204:         # Update  x.
205: 
206:         denom = 1.0/gamma
207:         w1 = w2
208:         w2 = w
209:         w = (v - oldeps*w1 - delta*w2) * denom
210:         x = x + phi*w
211: 
212:         # Go round again.
213: 
214:         gmax = max(gmax, gamma)
215:         gmin = min(gmin, gamma)
216:         z = rhs1 / gamma
217:         ynorm2 = z**2 + ynorm2
218:         rhs1 = rhs2 - delta*z
219:         rhs2 = - epsln*z
220: 
221:         # Estimate various norms and test for convergence.
222: 
223:         Anorm = sqrt(tnorm2)
224:         ynorm = sqrt(ynorm2)
225:         epsa = Anorm * eps
226:         epsx = Anorm * ynorm * eps
227:         epsr = Anorm * ynorm * tol
228:         diag = gbar
229: 
230:         if diag == 0:
231:             diag = epsa
232: 
233:         qrnorm = phibar
234:         rnorm = qrnorm
235:         test1 = rnorm / (Anorm*ynorm)    # ||r||  / (||A|| ||x||)
236:         test2 = root / Anorm            # ||Ar|| / (||A|| ||r||)
237: 
238:         # Estimate  cond(A).
239:         # In this version we look at the diagonals of  R  in the
240:         # factorization of the lower Hessenberg matrix,  Q * H = R,
241:         # where H is the tridiagonal matrix from Lanczos with one
242:         # extra row, beta(k+1) e_k^T.
243: 
244:         Acond = gmax/gmin
245: 
246:         # See if any of the stopping criteria are satisfied.
247:         # In rare cases, istop is already -1 from above (Abar = const*I).
248: 
249:         if istop == 0:
250:             t1 = 1 + test1      # These tests work if tol < eps
251:             t2 = 1 + test2
252:             if t2 <= 1:
253:                 istop = 2
254:             if t1 <= 1:
255:                 istop = 1
256: 
257:             if itn >= maxiter:
258:                 istop = 6
259:             if Acond >= 0.1/eps:
260:                 istop = 4
261:             if epsx >= beta:
262:                 istop = 3
263:             # if rnorm <= epsx   : istop = 2
264:             # if rnorm <= epsr   : istop = 1
265:             if test2 <= tol:
266:                 istop = 2
267:             if test1 <= tol:
268:                 istop = 1
269: 
270:         # See if it is time to print something.
271: 
272:         prnt = False
273:         if n <= 40:
274:             prnt = True
275:         if itn <= 10:
276:             prnt = True
277:         if itn >= maxiter-10:
278:             prnt = True
279:         if itn % 10 == 0:
280:             prnt = True
281:         if qrnorm <= 10*epsx:
282:             prnt = True
283:         if qrnorm <= 10*epsr:
284:             prnt = True
285:         if Acond <= 1e-2/eps:
286:             prnt = True
287:         if istop != 0:
288:             prnt = True
289: 
290:         if show and prnt:
291:             str1 = '%6g %12.5e %10.3e' % (itn, x[0], test1)
292:             str2 = ' %10.3e' % (test2,)
293:             str3 = ' %8.1e %8.1e %8.1e' % (Anorm, Acond, gbar/Anorm)
294: 
295:             print(str1 + str2 + str3)
296: 
297:             if itn % 10 == 0:
298:                 print()
299: 
300:         if callback is not None:
301:             callback(x)
302: 
303:         if istop != 0:
304:             break  # TODO check this
305: 
306:     if show:
307:         print()
308:         print(last + ' istop   =  %3g               itn   =%5g' % (istop,itn))
309:         print(last + ' Anorm   =  %12.4e      Acond =  %12.4e' % (Anorm,Acond))
310:         print(last + ' rnorm   =  %12.4e      ynorm =  %12.4e' % (rnorm,ynorm))
311:         print(last + ' Arnorm  =  %12.4e' % (Arnorm,))
312:         print(last + msg[istop+1])
313: 
314:     if istop == 6:
315:         info = maxiter
316:     else:
317:         info = 0
318: 
319:     return (postprocess(x),info)
320: 
321: 
322: if __name__ == '__main__':
323:     from scipy import ones, arange
324:     from scipy.linalg import norm
325:     from scipy.sparse import spdiags
326: 
327:     n = 10
328: 
329:     residuals = []
330: 
331:     def cb(x):
332:         residuals.append(norm(b - A*x))
333: 
334:     # A = poisson((10,),format='csr')
335:     A = spdiags([arange(1,n+1,dtype=float)], [0], n, n, format='csr')
336:     M = spdiags([1.0/arange(1,n+1,dtype=float)], [0], n, n, format='csr')
337:     A.psolve = M.matvec
338:     b = 0*ones(A.shape[0])
339:     x = minres(A,b,tol=1e-12,maxiter=None,callback=cb)
340:     # x = cg(A,b,x0=b,tol=1e-12,maxiter=None,callback=cb)[0]
341: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy import sqrt, inner, finfo, zeros' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_413365 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_413365) is not StypyTypeError):

    if (import_413365 != 'pyd_module'):
        __import__(import_413365)
        sys_modules_413366 = sys.modules[import_413365]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', sys_modules_413366.module_type_store, module_type_store, ['sqrt', 'inner', 'finfo', 'zeros'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_413366, sys_modules_413366.module_type_store, module_type_store)
    else:
        from numpy import sqrt, inner, finfo, zeros

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', None, module_type_store, ['sqrt', 'inner', 'finfo', 'zeros'], [sqrt, inner, finfo, zeros])

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_413365)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.linalg import norm' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_413367 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg')

if (type(import_413367) is not StypyTypeError):

    if (import_413367 != 'pyd_module'):
        __import__(import_413367)
        sys_modules_413368 = sys.modules[import_413367]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', sys_modules_413368.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_413368, sys_modules_413368.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', import_413367)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse.linalg.isolve.utils import make_system' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_413369 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg.isolve.utils')

if (type(import_413369) is not StypyTypeError):

    if (import_413369 != 'pyd_module'):
        __import__(import_413369)
        sys_modules_413370 = sys.modules[import_413369]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg.isolve.utils', sys_modules_413370.module_type_store, module_type_store, ['make_system'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_413370, sys_modules_413370.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.utils import make_system

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg.isolve.utils', None, module_type_store, ['make_system'], [make_system])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.utils' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg.isolve.utils', import_413369)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse.linalg.isolve.iterative import set_docstring' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_413371 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve.iterative')

if (type(import_413371) is not StypyTypeError):

    if (import_413371 != 'pyd_module'):
        __import__(import_413371)
        sys_modules_413372 = sys.modules[import_413371]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve.iterative', sys_modules_413372.module_type_store, module_type_store, ['set_docstring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_413372, sys_modules_413372.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.iterative import set_docstring

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve.iterative', None, module_type_store, ['set_docstring'], [set_docstring])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.iterative' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve.iterative', import_413371)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


# Assigning a List to a Name (line 9):

# Assigning a List to a Name (line 9):
__all__ = ['minres']
module_type_store.set_exportable_members(['minres'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_413373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_413374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'minres')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_413373, str_413374)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_413373)

# Assigning a Str to a Name (line 12):

# Assigning a Str to a Name (line 12):
str_413375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', 'Use MINimum RESidual iteration to solve Ax=b\n\nMINRES minimizes norm(A*x - b) for a real symmetric matrix A.  Unlike\nthe Conjugate Gradient method, A can be indefinite or singular.\n\nIf shift != 0 then the method solves (A - shift*I)x = b\n')
# Assigning a type to the variable 'header' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'header', str_413375)

# Assigning a Str to a Name (line 21):

# Assigning a Str to a Name (line 21):
str_413376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'The real symmetric N-by-N matrix of the linear system')
# Assigning a type to the variable 'Ainfo' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'Ainfo', str_413376)

# Assigning a Str to a Name (line 23):

# Assigning a Str to a Name (line 23):
str_413377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', '\nNotes\n-----\nTHIS FUNCTION IS EXPERIMENTAL AND SUBJECT TO CHANGE!\n\nReferences\n----------\nSolution of sparse indefinite systems of linear equations,\n    C. C. Paige and M. A. Saunders (1975),\n    SIAM J. Numer. Anal. 12(4), pp. 617-629.\n    https://web.stanford.edu/group/SOL/software/minres/\n\nThis file is a translation of the following MATLAB implementation:\n    https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip\n')
# Assigning a type to the variable 'footer' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'footer', str_413377)

@norecursion
def minres(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 44)
    None_413378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'None')
    float_413379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 32), 'float')
    float_413380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'float')
    # Getting the type of 'None' (line 44)
    None_413381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 55), 'None')
    # Getting the type of 'None' (line 45)
    None_413382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'None')
    # Getting the type of 'None' (line 45)
    None_413383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 28), 'None')
    # Getting the type of 'False' (line 45)
    False_413384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'False')
    # Getting the type of 'False' (line 45)
    False_413385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 52), 'False')
    defaults = [None_413378, float_413379, float_413380, None_413381, None_413382, None_413383, False_413384, False_413385]
    # Create a new context for function 'minres'
    module_type_store = module_type_store.open_function_context('minres', 41, 0, False)
    
    # Passed parameters checking function
    minres.stypy_localization = localization
    minres.stypy_type_of_self = None
    minres.stypy_type_store = module_type_store
    minres.stypy_function_name = 'minres'
    minres.stypy_param_names_list = ['A', 'b', 'x0', 'shift', 'tol', 'maxiter', 'M', 'callback', 'show', 'check']
    minres.stypy_varargs_param_name = None
    minres.stypy_kwargs_param_name = None
    minres.stypy_call_defaults = defaults
    minres.stypy_call_varargs = varargs
    minres.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minres', ['A', 'b', 'x0', 'shift', 'tol', 'maxiter', 'M', 'callback', 'show', 'check'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minres', localization, ['A', 'b', 'x0', 'shift', 'tol', 'maxiter', 'M', 'callback', 'show', 'check'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minres(...)' code ##################

    
    # Assigning a Call to a Tuple (line 46):
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_413386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    
    # Call to make_system(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'A' (line 46)
    A_413388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'A', False)
    # Getting the type of 'M' (line 46)
    M_413389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'M', False)
    # Getting the type of 'x0' (line 46)
    x0_413390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 48), 'x0', False)
    # Getting the type of 'b' (line 46)
    b_413391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 52), 'b', False)
    # Processing the call keyword arguments (line 46)
    kwargs_413392 = {}
    # Getting the type of 'make_system' (line 46)
    make_system_413387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 46)
    make_system_call_result_413393 = invoke(stypy.reporting.localization.Localization(__file__, 46, 30), make_system_413387, *[A_413388, M_413389, x0_413390, b_413391], **kwargs_413392)
    
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___413394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), make_system_call_result_413393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_413395 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___413394, int_413386)
    
    # Assigning a type to the variable 'tuple_var_assignment_413360' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413360', subscript_call_result_413395)
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_413396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    
    # Call to make_system(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'A' (line 46)
    A_413398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'A', False)
    # Getting the type of 'M' (line 46)
    M_413399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'M', False)
    # Getting the type of 'x0' (line 46)
    x0_413400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 48), 'x0', False)
    # Getting the type of 'b' (line 46)
    b_413401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 52), 'b', False)
    # Processing the call keyword arguments (line 46)
    kwargs_413402 = {}
    # Getting the type of 'make_system' (line 46)
    make_system_413397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 46)
    make_system_call_result_413403 = invoke(stypy.reporting.localization.Localization(__file__, 46, 30), make_system_413397, *[A_413398, M_413399, x0_413400, b_413401], **kwargs_413402)
    
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___413404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), make_system_call_result_413403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_413405 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___413404, int_413396)
    
    # Assigning a type to the variable 'tuple_var_assignment_413361' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413361', subscript_call_result_413405)
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_413406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    
    # Call to make_system(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'A' (line 46)
    A_413408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'A', False)
    # Getting the type of 'M' (line 46)
    M_413409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'M', False)
    # Getting the type of 'x0' (line 46)
    x0_413410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 48), 'x0', False)
    # Getting the type of 'b' (line 46)
    b_413411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 52), 'b', False)
    # Processing the call keyword arguments (line 46)
    kwargs_413412 = {}
    # Getting the type of 'make_system' (line 46)
    make_system_413407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 46)
    make_system_call_result_413413 = invoke(stypy.reporting.localization.Localization(__file__, 46, 30), make_system_413407, *[A_413408, M_413409, x0_413410, b_413411], **kwargs_413412)
    
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___413414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), make_system_call_result_413413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_413415 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___413414, int_413406)
    
    # Assigning a type to the variable 'tuple_var_assignment_413362' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413362', subscript_call_result_413415)
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_413416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    
    # Call to make_system(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'A' (line 46)
    A_413418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'A', False)
    # Getting the type of 'M' (line 46)
    M_413419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'M', False)
    # Getting the type of 'x0' (line 46)
    x0_413420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 48), 'x0', False)
    # Getting the type of 'b' (line 46)
    b_413421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 52), 'b', False)
    # Processing the call keyword arguments (line 46)
    kwargs_413422 = {}
    # Getting the type of 'make_system' (line 46)
    make_system_413417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 46)
    make_system_call_result_413423 = invoke(stypy.reporting.localization.Localization(__file__, 46, 30), make_system_413417, *[A_413418, M_413419, x0_413420, b_413421], **kwargs_413422)
    
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___413424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), make_system_call_result_413423, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_413425 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___413424, int_413416)
    
    # Assigning a type to the variable 'tuple_var_assignment_413363' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413363', subscript_call_result_413425)
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_413426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    
    # Call to make_system(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'A' (line 46)
    A_413428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'A', False)
    # Getting the type of 'M' (line 46)
    M_413429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'M', False)
    # Getting the type of 'x0' (line 46)
    x0_413430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 48), 'x0', False)
    # Getting the type of 'b' (line 46)
    b_413431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 52), 'b', False)
    # Processing the call keyword arguments (line 46)
    kwargs_413432 = {}
    # Getting the type of 'make_system' (line 46)
    make_system_413427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'make_system', False)
    # Calling make_system(args, kwargs) (line 46)
    make_system_call_result_413433 = invoke(stypy.reporting.localization.Localization(__file__, 46, 30), make_system_413427, *[A_413428, M_413429, x0_413430, b_413431], **kwargs_413432)
    
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___413434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), make_system_call_result_413433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_413435 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___413434, int_413426)
    
    # Assigning a type to the variable 'tuple_var_assignment_413364' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413364', subscript_call_result_413435)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_413360' (line 46)
    tuple_var_assignment_413360_413436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413360')
    # Assigning a type to the variable 'A' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'A', tuple_var_assignment_413360_413436)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_413361' (line 46)
    tuple_var_assignment_413361_413437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413361')
    # Assigning a type to the variable 'M' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'M', tuple_var_assignment_413361_413437)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_413362' (line 46)
    tuple_var_assignment_413362_413438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413362')
    # Assigning a type to the variable 'x' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 10), 'x', tuple_var_assignment_413362_413438)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_413363' (line 46)
    tuple_var_assignment_413363_413439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413363')
    # Assigning a type to the variable 'b' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'b', tuple_var_assignment_413363_413439)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_413364' (line 46)
    tuple_var_assignment_413364_413440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_413364')
    # Assigning a type to the variable 'postprocess' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'postprocess', tuple_var_assignment_413364_413440)
    
    # Assigning a Attribute to a Name (line 48):
    
    # Assigning a Attribute to a Name (line 48):
    # Getting the type of 'A' (line 48)
    A_413441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 13), 'A')
    # Obtaining the member 'matvec' of a type (line 48)
    matvec_413442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 13), A_413441, 'matvec')
    # Assigning a type to the variable 'matvec' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'matvec', matvec_413442)
    
    # Assigning a Attribute to a Name (line 49):
    
    # Assigning a Attribute to a Name (line 49):
    # Getting the type of 'M' (line 49)
    M_413443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'M')
    # Obtaining the member 'matvec' of a type (line 49)
    matvec_413444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 13), M_413443, 'matvec')
    # Assigning a type to the variable 'psolve' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'psolve', matvec_413444)
    
    # Assigning a Str to a Name (line 51):
    
    # Assigning a Str to a Name (line 51):
    str_413445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'str', 'Enter minres.   ')
    # Assigning a type to the variable 'first' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'first', str_413445)
    
    # Assigning a Str to a Name (line 52):
    
    # Assigning a Str to a Name (line 52):
    str_413446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 11), 'str', 'Exit  minres.   ')
    # Assigning a type to the variable 'last' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'last', str_413446)
    
    # Assigning a Subscript to a Name (line 54):
    
    # Assigning a Subscript to a Name (line 54):
    
    # Obtaining the type of the subscript
    int_413447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'int')
    # Getting the type of 'A' (line 54)
    A_413448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'A')
    # Obtaining the member 'shape' of a type (line 54)
    shape_413449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), A_413448, 'shape')
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___413450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), shape_413449, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_413451 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___413450, int_413447)
    
    # Assigning a type to the variable 'n' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'n', subscript_call_result_413451)
    
    # Type idiom detected: calculating its left and rigth part (line 56)
    # Getting the type of 'maxiter' (line 56)
    maxiter_413452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 7), 'maxiter')
    # Getting the type of 'None' (line 56)
    None_413453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'None')
    
    (may_be_413454, more_types_in_union_413455) = may_be_none(maxiter_413452, None_413453)

    if may_be_413454:

        if more_types_in_union_413455:
            # Runtime conditional SSA (line 56)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 57):
        
        # Assigning a BinOp to a Name (line 57):
        int_413456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'int')
        # Getting the type of 'n' (line 57)
        n_413457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 22), 'n')
        # Applying the binary operator '*' (line 57)
        result_mul_413458 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 18), '*', int_413456, n_413457)
        
        # Assigning a type to the variable 'maxiter' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'maxiter', result_mul_413458)

        if more_types_in_union_413455:
            # SSA join for if statement (line 56)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 59):
    
    # Assigning a List to a Name (line 59):
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_413459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    str_413460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 11), 'str', ' beta2 = 0.  If M = I, b and x are eigenvectors    ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413460)
    # Adding element type (line 59)
    str_413461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'str', ' beta1 = 0.  The exact solution is  x = 0          ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413461)
    # Adding element type (line 59)
    str_413462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'str', ' A solution to Ax = b was found, given rtol        ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413462)
    # Adding element type (line 59)
    str_413463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'str', ' A least-squares solution was found, given rtol    ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413463)
    # Adding element type (line 59)
    str_413464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'str', ' Reasonable accuracy achieved, given eps           ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413464)
    # Adding element type (line 59)
    str_413465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'str', ' x has converged to an eigenvector                 ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413465)
    # Adding element type (line 59)
    str_413466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'str', ' acond has exceeded 0.1/eps                        ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413466)
    # Adding element type (line 59)
    str_413467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'str', ' The iteration limit was reached                   ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413467)
    # Adding element type (line 59)
    str_413468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 12), 'str', ' A  does not define a symmetric matrix             ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413468)
    # Adding element type (line 59)
    str_413469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'str', ' M  does not define a symmetric matrix             ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413469)
    # Adding element type (line 59)
    str_413470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 12), 'str', ' M  does not define a pos-def preconditioner       ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_413459, str_413470)
    
    # Assigning a type to the variable 'msg' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'msg', list_413459)
    
    # Getting the type of 'show' (line 71)
    show_413471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'show')
    # Testing the type of an if condition (line 71)
    if_condition_413472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), show_413471)
    # Assigning a type to the variable 'if_condition_413472' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_413472', if_condition_413472)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'first' (line 72)
    first_413474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'first', False)
    str_413475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'str', 'Solution of symmetric Ax = b')
    # Applying the binary operator '+' (line 72)
    result_add_413476 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 14), '+', first_413474, str_413475)
    
    # Processing the call keyword arguments (line 72)
    kwargs_413477 = {}
    # Getting the type of 'print' (line 72)
    print_413473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'print', False)
    # Calling print(args, kwargs) (line 72)
    print_call_result_413478 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), print_413473, *[result_add_413476], **kwargs_413477)
    
    
    # Call to print(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'first' (line 73)
    first_413480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'first', False)
    str_413481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'str', 'n      =  %3g     shift  =  %23.14e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 73)
    tuple_413482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 63), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 73)
    # Adding element type (line 73)
    # Getting the type of 'n' (line 73)
    n_413483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 63), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 63), tuple_413482, n_413483)
    # Adding element type (line 73)
    # Getting the type of 'shift' (line 73)
    shift_413484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 65), 'shift', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 63), tuple_413482, shift_413484)
    
    # Applying the binary operator '%' (line 73)
    result_mod_413485 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 22), '%', str_413481, tuple_413482)
    
    # Applying the binary operator '+' (line 73)
    result_add_413486 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 14), '+', first_413480, result_mod_413485)
    
    # Processing the call keyword arguments (line 73)
    kwargs_413487 = {}
    # Getting the type of 'print' (line 73)
    print_413479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'print', False)
    # Calling print(args, kwargs) (line 73)
    print_call_result_413488 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), print_413479, *[result_add_413486], **kwargs_413487)
    
    
    # Call to print(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'first' (line 74)
    first_413490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'first', False)
    str_413491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 22), 'str', 'itnlim =  %3g     rtol   =  %11.2e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 74)
    tuple_413492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 62), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 74)
    # Adding element type (line 74)
    # Getting the type of 'maxiter' (line 74)
    maxiter_413493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 62), 'maxiter', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 62), tuple_413492, maxiter_413493)
    # Adding element type (line 74)
    # Getting the type of 'tol' (line 74)
    tol_413494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 70), 'tol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 62), tuple_413492, tol_413494)
    
    # Applying the binary operator '%' (line 74)
    result_mod_413495 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 22), '%', str_413491, tuple_413492)
    
    # Applying the binary operator '+' (line 74)
    result_add_413496 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 14), '+', first_413490, result_mod_413495)
    
    # Processing the call keyword arguments (line 74)
    kwargs_413497 = {}
    # Getting the type of 'print' (line 74)
    print_413489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'print', False)
    # Calling print(args, kwargs) (line 74)
    print_call_result_413498 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), print_413489, *[result_add_413496], **kwargs_413497)
    
    
    # Call to print(...): (line 75)
    # Processing the call keyword arguments (line 75)
    kwargs_413500 = {}
    # Getting the type of 'print' (line 75)
    print_413499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'print', False)
    # Calling print(args, kwargs) (line 75)
    print_call_result_413501 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), print_413499, *[], **kwargs_413500)
    
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 77):
    
    # Assigning a Num to a Name (line 77):
    int_413502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 12), 'int')
    # Assigning a type to the variable 'istop' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'istop', int_413502)
    
    # Assigning a Num to a Name (line 78):
    
    # Assigning a Num to a Name (line 78):
    int_413503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 10), 'int')
    # Assigning a type to the variable 'itn' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'itn', int_413503)
    
    # Assigning a Num to a Name (line 79):
    
    # Assigning a Num to a Name (line 79):
    int_413504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 12), 'int')
    # Assigning a type to the variable 'Anorm' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'Anorm', int_413504)
    
    # Assigning a Num to a Name (line 80):
    
    # Assigning a Num to a Name (line 80):
    int_413505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
    # Assigning a type to the variable 'Acond' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'Acond', int_413505)
    
    # Assigning a Num to a Name (line 81):
    
    # Assigning a Num to a Name (line 81):
    int_413506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 12), 'int')
    # Assigning a type to the variable 'rnorm' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'rnorm', int_413506)
    
    # Assigning a Num to a Name (line 82):
    
    # Assigning a Num to a Name (line 82):
    int_413507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'int')
    # Assigning a type to the variable 'ynorm' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'ynorm', int_413507)
    
    # Assigning a Attribute to a Name (line 84):
    
    # Assigning a Attribute to a Name (line 84):
    # Getting the type of 'x' (line 84)
    x_413508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'x')
    # Obtaining the member 'dtype' of a type (line 84)
    dtype_413509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), x_413508, 'dtype')
    # Assigning a type to the variable 'xtype' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'xtype', dtype_413509)
    
    # Assigning a Attribute to a Name (line 86):
    
    # Assigning a Attribute to a Name (line 86):
    
    # Call to finfo(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'xtype' (line 86)
    xtype_413511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'xtype', False)
    # Processing the call keyword arguments (line 86)
    kwargs_413512 = {}
    # Getting the type of 'finfo' (line 86)
    finfo_413510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 10), 'finfo', False)
    # Calling finfo(args, kwargs) (line 86)
    finfo_call_result_413513 = invoke(stypy.reporting.localization.Localization(__file__, 86, 10), finfo_413510, *[xtype_413511], **kwargs_413512)
    
    # Obtaining the member 'eps' of a type (line 86)
    eps_413514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 10), finfo_call_result_413513, 'eps')
    # Assigning a type to the variable 'eps' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'eps', eps_413514)
    
    # Assigning a Call to a Name (line 88):
    
    # Assigning a Call to a Name (line 88):
    
    # Call to zeros(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'n' (line 88)
    n_413516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'n', False)
    # Processing the call keyword arguments (line 88)
    # Getting the type of 'xtype' (line 88)
    xtype_413517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'xtype', False)
    keyword_413518 = xtype_413517
    kwargs_413519 = {'dtype': keyword_413518}
    # Getting the type of 'zeros' (line 88)
    zeros_413515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 88)
    zeros_call_result_413520 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), zeros_413515, *[n_413516], **kwargs_413519)
    
    # Assigning a type to the variable 'x' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'x', zeros_call_result_413520)
    
    # Assigning a Name to a Name (line 94):
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'b' (line 94)
    b_413521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'b')
    # Assigning a type to the variable 'y' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'y', b_413521)
    
    # Assigning a Name to a Name (line 95):
    
    # Assigning a Name to a Name (line 95):
    # Getting the type of 'b' (line 95)
    b_413522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 9), 'b')
    # Assigning a type to the variable 'r1' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'r1', b_413522)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to psolve(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'b' (line 97)
    b_413524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'b', False)
    # Processing the call keyword arguments (line 97)
    kwargs_413525 = {}
    # Getting the type of 'psolve' (line 97)
    psolve_413523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'psolve', False)
    # Calling psolve(args, kwargs) (line 97)
    psolve_call_result_413526 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), psolve_413523, *[b_413524], **kwargs_413525)
    
    # Assigning a type to the variable 'y' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'y', psolve_call_result_413526)
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to inner(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'b' (line 99)
    b_413528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'b', False)
    # Getting the type of 'y' (line 99)
    y_413529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'y', False)
    # Processing the call keyword arguments (line 99)
    kwargs_413530 = {}
    # Getting the type of 'inner' (line 99)
    inner_413527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'inner', False)
    # Calling inner(args, kwargs) (line 99)
    inner_call_result_413531 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), inner_413527, *[b_413528, y_413529], **kwargs_413530)
    
    # Assigning a type to the variable 'beta1' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'beta1', inner_call_result_413531)
    
    
    # Getting the type of 'beta1' (line 101)
    beta1_413532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'beta1')
    int_413533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'int')
    # Applying the binary operator '<' (line 101)
    result_lt_413534 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), '<', beta1_413532, int_413533)
    
    # Testing the type of an if condition (line 101)
    if_condition_413535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), result_lt_413534)
    # Assigning a type to the variable 'if_condition_413535' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_413535', if_condition_413535)
    # SSA begins for if statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 102)
    # Processing the call arguments (line 102)
    str_413537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 25), 'str', 'indefinite preconditioner')
    # Processing the call keyword arguments (line 102)
    kwargs_413538 = {}
    # Getting the type of 'ValueError' (line 102)
    ValueError_413536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 102)
    ValueError_call_result_413539 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), ValueError_413536, *[str_413537], **kwargs_413538)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 102, 8), ValueError_call_result_413539, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 101)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'beta1' (line 103)
    beta1_413540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 9), 'beta1')
    int_413541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'int')
    # Applying the binary operator '==' (line 103)
    result_eq_413542 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 9), '==', beta1_413540, int_413541)
    
    # Testing the type of an if condition (line 103)
    if_condition_413543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 9), result_eq_413542)
    # Assigning a type to the variable 'if_condition_413543' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 9), 'if_condition_413543', if_condition_413543)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_413544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    
    # Call to postprocess(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'x' (line 104)
    x_413546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'x', False)
    # Processing the call keyword arguments (line 104)
    kwargs_413547 = {}
    # Getting the type of 'postprocess' (line 104)
    postprocess_413545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 104)
    postprocess_call_result_413548 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), postprocess_413545, *[x_413546], **kwargs_413547)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 16), tuple_413544, postprocess_call_result_413548)
    # Adding element type (line 104)
    int_413549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 16), tuple_413544, int_413549)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', tuple_413544)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 101)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to sqrt(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'beta1' (line 106)
    beta1_413551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'beta1', False)
    # Processing the call keyword arguments (line 106)
    kwargs_413552 = {}
    # Getting the type of 'sqrt' (line 106)
    sqrt_413550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 106)
    sqrt_call_result_413553 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), sqrt_413550, *[beta1_413551], **kwargs_413552)
    
    # Assigning a type to the variable 'beta1' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'beta1', sqrt_call_result_413553)
    
    # Getting the type of 'check' (line 108)
    check_413554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'check')
    # Testing the type of an if condition (line 108)
    if_condition_413555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), check_413554)
    # Assigning a type to the variable 'if_condition_413555' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_413555', if_condition_413555)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to matvec(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'y' (line 112)
    y_413557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'y', False)
    # Processing the call keyword arguments (line 112)
    kwargs_413558 = {}
    # Getting the type of 'matvec' (line 112)
    matvec_413556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'matvec', False)
    # Calling matvec(args, kwargs) (line 112)
    matvec_call_result_413559 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), matvec_413556, *[y_413557], **kwargs_413558)
    
    # Assigning a type to the variable 'w' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'w', matvec_call_result_413559)
    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to matvec(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'w' (line 113)
    w_413561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'w', False)
    # Processing the call keyword arguments (line 113)
    kwargs_413562 = {}
    # Getting the type of 'matvec' (line 113)
    matvec_413560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'matvec', False)
    # Calling matvec(args, kwargs) (line 113)
    matvec_call_result_413563 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), matvec_413560, *[w_413561], **kwargs_413562)
    
    # Assigning a type to the variable 'r2' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'r2', matvec_call_result_413563)
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to inner(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'w' (line 114)
    w_413565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'w', False)
    # Getting the type of 'w' (line 114)
    w_413566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'w', False)
    # Processing the call keyword arguments (line 114)
    kwargs_413567 = {}
    # Getting the type of 'inner' (line 114)
    inner_413564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'inner', False)
    # Calling inner(args, kwargs) (line 114)
    inner_call_result_413568 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), inner_413564, *[w_413565, w_413566], **kwargs_413567)
    
    # Assigning a type to the variable 's' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 's', inner_call_result_413568)
    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to inner(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'y' (line 115)
    y_413570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'y', False)
    # Getting the type of 'r2' (line 115)
    r2_413571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'r2', False)
    # Processing the call keyword arguments (line 115)
    kwargs_413572 = {}
    # Getting the type of 'inner' (line 115)
    inner_413569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'inner', False)
    # Calling inner(args, kwargs) (line 115)
    inner_call_result_413573 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), inner_413569, *[y_413570, r2_413571], **kwargs_413572)
    
    # Assigning a type to the variable 't' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 't', inner_call_result_413573)
    
    # Assigning a Call to a Name (line 116):
    
    # Assigning a Call to a Name (line 116):
    
    # Call to abs(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 's' (line 116)
    s_413575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 's', False)
    # Getting the type of 't' (line 116)
    t_413576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 't', False)
    # Applying the binary operator '-' (line 116)
    result_sub_413577 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 16), '-', s_413575, t_413576)
    
    # Processing the call keyword arguments (line 116)
    kwargs_413578 = {}
    # Getting the type of 'abs' (line 116)
    abs_413574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'abs', False)
    # Calling abs(args, kwargs) (line 116)
    abs_call_result_413579 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), abs_413574, *[result_sub_413577], **kwargs_413578)
    
    # Assigning a type to the variable 'z' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'z', abs_call_result_413579)
    
    # Assigning a BinOp to a Name (line 117):
    
    # Assigning a BinOp to a Name (line 117):
    # Getting the type of 's' (line 117)
    s_413580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 's')
    # Getting the type of 'eps' (line 117)
    eps_413581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'eps')
    # Applying the binary operator '+' (line 117)
    result_add_413582 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 16), '+', s_413580, eps_413581)
    
    # Getting the type of 'eps' (line 117)
    eps_413583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'eps')
    float_413584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'float')
    float_413585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 37), 'float')
    # Applying the binary operator 'div' (line 117)
    result_div_413586 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 33), 'div', float_413584, float_413585)
    
    # Applying the binary operator '**' (line 117)
    result_pow_413587 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 27), '**', eps_413583, result_div_413586)
    
    # Applying the binary operator '*' (line 117)
    result_mul_413588 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 15), '*', result_add_413582, result_pow_413587)
    
    # Assigning a type to the variable 'epsa' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'epsa', result_mul_413588)
    
    
    # Getting the type of 'z' (line 118)
    z_413589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'z')
    # Getting the type of 'epsa' (line 118)
    epsa_413590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'epsa')
    # Applying the binary operator '>' (line 118)
    result_gt_413591 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 11), '>', z_413589, epsa_413590)
    
    # Testing the type of an if condition (line 118)
    if_condition_413592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 8), result_gt_413591)
    # Assigning a type to the variable 'if_condition_413592' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'if_condition_413592', if_condition_413592)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 119)
    # Processing the call arguments (line 119)
    str_413594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 29), 'str', 'non-symmetric matrix')
    # Processing the call keyword arguments (line 119)
    kwargs_413595 = {}
    # Getting the type of 'ValueError' (line 119)
    ValueError_413593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 119)
    ValueError_call_result_413596 = invoke(stypy.reporting.localization.Localization(__file__, 119, 18), ValueError_413593, *[str_413594], **kwargs_413595)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 119, 12), ValueError_call_result_413596, 'raise parameter', BaseException)
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to psolve(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'y' (line 122)
    y_413598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'y', False)
    # Processing the call keyword arguments (line 122)
    kwargs_413599 = {}
    # Getting the type of 'psolve' (line 122)
    psolve_413597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'psolve', False)
    # Calling psolve(args, kwargs) (line 122)
    psolve_call_result_413600 = invoke(stypy.reporting.localization.Localization(__file__, 122, 13), psolve_413597, *[y_413598], **kwargs_413599)
    
    # Assigning a type to the variable 'r2' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'r2', psolve_call_result_413600)
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to inner(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'y' (line 123)
    y_413602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'y', False)
    # Getting the type of 'y' (line 123)
    y_413603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'y', False)
    # Processing the call keyword arguments (line 123)
    kwargs_413604 = {}
    # Getting the type of 'inner' (line 123)
    inner_413601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'inner', False)
    # Calling inner(args, kwargs) (line 123)
    inner_call_result_413605 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), inner_413601, *[y_413602, y_413603], **kwargs_413604)
    
    # Assigning a type to the variable 's' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 's', inner_call_result_413605)
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to inner(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'r1' (line 124)
    r1_413607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'r1', False)
    # Getting the type of 'r2' (line 124)
    r2_413608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'r2', False)
    # Processing the call keyword arguments (line 124)
    kwargs_413609 = {}
    # Getting the type of 'inner' (line 124)
    inner_413606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'inner', False)
    # Calling inner(args, kwargs) (line 124)
    inner_call_result_413610 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), inner_413606, *[r1_413607, r2_413608], **kwargs_413609)
    
    # Assigning a type to the variable 't' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 't', inner_call_result_413610)
    
    # Assigning a Call to a Name (line 125):
    
    # Assigning a Call to a Name (line 125):
    
    # Call to abs(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 's' (line 125)
    s_413612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 's', False)
    # Getting the type of 't' (line 125)
    t_413613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 't', False)
    # Applying the binary operator '-' (line 125)
    result_sub_413614 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 16), '-', s_413612, t_413613)
    
    # Processing the call keyword arguments (line 125)
    kwargs_413615 = {}
    # Getting the type of 'abs' (line 125)
    abs_413611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'abs', False)
    # Calling abs(args, kwargs) (line 125)
    abs_call_result_413616 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), abs_413611, *[result_sub_413614], **kwargs_413615)
    
    # Assigning a type to the variable 'z' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'z', abs_call_result_413616)
    
    # Assigning a BinOp to a Name (line 126):
    
    # Assigning a BinOp to a Name (line 126):
    # Getting the type of 's' (line 126)
    s_413617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 's')
    # Getting the type of 'eps' (line 126)
    eps_413618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'eps')
    # Applying the binary operator '+' (line 126)
    result_add_413619 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 16), '+', s_413617, eps_413618)
    
    # Getting the type of 'eps' (line 126)
    eps_413620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'eps')
    float_413621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 33), 'float')
    float_413622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 37), 'float')
    # Applying the binary operator 'div' (line 126)
    result_div_413623 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 33), 'div', float_413621, float_413622)
    
    # Applying the binary operator '**' (line 126)
    result_pow_413624 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 27), '**', eps_413620, result_div_413623)
    
    # Applying the binary operator '*' (line 126)
    result_mul_413625 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), '*', result_add_413619, result_pow_413624)
    
    # Assigning a type to the variable 'epsa' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'epsa', result_mul_413625)
    
    
    # Getting the type of 'z' (line 127)
    z_413626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'z')
    # Getting the type of 'epsa' (line 127)
    epsa_413627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'epsa')
    # Applying the binary operator '>' (line 127)
    result_gt_413628 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 11), '>', z_413626, epsa_413627)
    
    # Testing the type of an if condition (line 127)
    if_condition_413629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 8), result_gt_413628)
    # Assigning a type to the variable 'if_condition_413629' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'if_condition_413629', if_condition_413629)
    # SSA begins for if statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 128)
    # Processing the call arguments (line 128)
    str_413631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 29), 'str', 'non-symmetric preconditioner')
    # Processing the call keyword arguments (line 128)
    kwargs_413632 = {}
    # Getting the type of 'ValueError' (line 128)
    ValueError_413630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 128)
    ValueError_call_result_413633 = invoke(stypy.reporting.localization.Localization(__file__, 128, 18), ValueError_413630, *[str_413631], **kwargs_413632)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 128, 12), ValueError_call_result_413633, 'raise parameter', BaseException)
    # SSA join for if statement (line 127)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 131):
    
    # Assigning a Num to a Name (line 131):
    int_413634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 11), 'int')
    # Assigning a type to the variable 'oldb' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'oldb', int_413634)
    
    # Assigning a Name to a Name (line 132):
    
    # Assigning a Name to a Name (line 132):
    # Getting the type of 'beta1' (line 132)
    beta1_413635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'beta1')
    # Assigning a type to the variable 'beta' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'beta', beta1_413635)
    
    # Assigning a Num to a Name (line 133):
    
    # Assigning a Num to a Name (line 133):
    int_413636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 11), 'int')
    # Assigning a type to the variable 'dbar' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'dbar', int_413636)
    
    # Assigning a Num to a Name (line 134):
    
    # Assigning a Num to a Name (line 134):
    int_413637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 12), 'int')
    # Assigning a type to the variable 'epsln' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'epsln', int_413637)
    
    # Assigning a Name to a Name (line 135):
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'beta1' (line 135)
    beta1_413638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'beta1')
    # Assigning a type to the variable 'qrnorm' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'qrnorm', beta1_413638)
    
    # Assigning a Name to a Name (line 136):
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'beta1' (line 136)
    beta1_413639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 13), 'beta1')
    # Assigning a type to the variable 'phibar' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'phibar', beta1_413639)
    
    # Assigning a Name to a Name (line 137):
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'beta1' (line 137)
    beta1_413640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'beta1')
    # Assigning a type to the variable 'rhs1' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'rhs1', beta1_413640)
    
    # Assigning a Num to a Name (line 138):
    
    # Assigning a Num to a Name (line 138):
    int_413641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 11), 'int')
    # Assigning a type to the variable 'rhs2' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'rhs2', int_413641)
    
    # Assigning a Num to a Name (line 139):
    
    # Assigning a Num to a Name (line 139):
    int_413642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 13), 'int')
    # Assigning a type to the variable 'tnorm2' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'tnorm2', int_413642)
    
    # Assigning a Num to a Name (line 140):
    
    # Assigning a Num to a Name (line 140):
    int_413643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 13), 'int')
    # Assigning a type to the variable 'ynorm2' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'ynorm2', int_413643)
    
    # Assigning a Num to a Name (line 141):
    
    # Assigning a Num to a Name (line 141):
    int_413644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 9), 'int')
    # Assigning a type to the variable 'cs' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'cs', int_413644)
    
    # Assigning a Num to a Name (line 142):
    
    # Assigning a Num to a Name (line 142):
    int_413645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 9), 'int')
    # Assigning a type to the variable 'sn' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'sn', int_413645)
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to zeros(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'n' (line 143)
    n_413647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), 'n', False)
    # Processing the call keyword arguments (line 143)
    # Getting the type of 'xtype' (line 143)
    xtype_413648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'xtype', False)
    keyword_413649 = xtype_413648
    kwargs_413650 = {'dtype': keyword_413649}
    # Getting the type of 'zeros' (line 143)
    zeros_413646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 143)
    zeros_call_result_413651 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), zeros_413646, *[n_413647], **kwargs_413650)
    
    # Assigning a type to the variable 'w' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'w', zeros_call_result_413651)
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to zeros(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'n' (line 144)
    n_413653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'n', False)
    # Processing the call keyword arguments (line 144)
    # Getting the type of 'xtype' (line 144)
    xtype_413654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'xtype', False)
    keyword_413655 = xtype_413654
    kwargs_413656 = {'dtype': keyword_413655}
    # Getting the type of 'zeros' (line 144)
    zeros_413652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 9), 'zeros', False)
    # Calling zeros(args, kwargs) (line 144)
    zeros_call_result_413657 = invoke(stypy.reporting.localization.Localization(__file__, 144, 9), zeros_413652, *[n_413653], **kwargs_413656)
    
    # Assigning a type to the variable 'w2' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'w2', zeros_call_result_413657)
    
    # Assigning a Name to a Name (line 145):
    
    # Assigning a Name to a Name (line 145):
    # Getting the type of 'r1' (line 145)
    r1_413658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 9), 'r1')
    # Assigning a type to the variable 'r2' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'r2', r1_413658)
    
    # Getting the type of 'show' (line 147)
    show_413659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'show')
    # Testing the type of an if condition (line 147)
    if_condition_413660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), show_413659)
    # Assigning a type to the variable 'if_condition_413660' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_413660', if_condition_413660)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_413662 = {}
    # Getting the type of 'print' (line 148)
    print_413661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'print', False)
    # Calling print(args, kwargs) (line 148)
    print_call_result_413663 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), print_413661, *[], **kwargs_413662)
    
    
    # Call to print(...): (line 149)
    # Processing the call keyword arguments (line 149)
    kwargs_413665 = {}
    # Getting the type of 'print' (line 149)
    print_413664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'print', False)
    # Calling print(args, kwargs) (line 149)
    print_call_result_413666 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), print_413664, *[], **kwargs_413665)
    
    
    # Call to print(...): (line 150)
    # Processing the call arguments (line 150)
    str_413668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 14), 'str', '   Itn     x(1)     Compatible    LS       norm(A)  cond(A) gbar/|A|')
    # Processing the call keyword arguments (line 150)
    kwargs_413669 = {}
    # Getting the type of 'print' (line 150)
    print_413667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'print', False)
    # Calling print(args, kwargs) (line 150)
    print_call_result_413670 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), print_413667, *[str_413668], **kwargs_413669)
    
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 152)
    itn_413671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 10), 'itn')
    # Getting the type of 'maxiter' (line 152)
    maxiter_413672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'maxiter')
    # Applying the binary operator '<' (line 152)
    result_lt_413673 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 10), '<', itn_413671, maxiter_413672)
    
    # Testing the type of an if condition (line 152)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 4), result_lt_413673)
    # SSA begins for while statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'itn' (line 153)
    itn_413674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'itn')
    int_413675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 15), 'int')
    # Applying the binary operator '+=' (line 153)
    result_iadd_413676 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 8), '+=', itn_413674, int_413675)
    # Assigning a type to the variable 'itn' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'itn', result_iadd_413676)
    
    
    # Assigning a BinOp to a Name (line 155):
    
    # Assigning a BinOp to a Name (line 155):
    float_413677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 12), 'float')
    # Getting the type of 'beta' (line 155)
    beta_413678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'beta')
    # Applying the binary operator 'div' (line 155)
    result_div_413679 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 12), 'div', float_413677, beta_413678)
    
    # Assigning a type to the variable 's' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 's', result_div_413679)
    
    # Assigning a BinOp to a Name (line 156):
    
    # Assigning a BinOp to a Name (line 156):
    # Getting the type of 's' (line 156)
    s_413680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 's')
    # Getting the type of 'y' (line 156)
    y_413681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'y')
    # Applying the binary operator '*' (line 156)
    result_mul_413682 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 12), '*', s_413680, y_413681)
    
    # Assigning a type to the variable 'v' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'v', result_mul_413682)
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to matvec(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'v' (line 158)
    v_413684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'v', False)
    # Processing the call keyword arguments (line 158)
    kwargs_413685 = {}
    # Getting the type of 'matvec' (line 158)
    matvec_413683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'matvec', False)
    # Calling matvec(args, kwargs) (line 158)
    matvec_call_result_413686 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), matvec_413683, *[v_413684], **kwargs_413685)
    
    # Assigning a type to the variable 'y' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'y', matvec_call_result_413686)
    
    # Assigning a BinOp to a Name (line 159):
    
    # Assigning a BinOp to a Name (line 159):
    # Getting the type of 'y' (line 159)
    y_413687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'y')
    # Getting the type of 'shift' (line 159)
    shift_413688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'shift')
    # Getting the type of 'v' (line 159)
    v_413689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'v')
    # Applying the binary operator '*' (line 159)
    result_mul_413690 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 16), '*', shift_413688, v_413689)
    
    # Applying the binary operator '-' (line 159)
    result_sub_413691 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 12), '-', y_413687, result_mul_413690)
    
    # Assigning a type to the variable 'y' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'y', result_sub_413691)
    
    
    # Getting the type of 'itn' (line 161)
    itn_413692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'itn')
    int_413693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'int')
    # Applying the binary operator '>=' (line 161)
    result_ge_413694 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), '>=', itn_413692, int_413693)
    
    # Testing the type of an if condition (line 161)
    if_condition_413695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_ge_413694)
    # Assigning a type to the variable 'if_condition_413695' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_413695', if_condition_413695)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 162):
    
    # Assigning a BinOp to a Name (line 162):
    # Getting the type of 'y' (line 162)
    y_413696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'y')
    # Getting the type of 'beta' (line 162)
    beta_413697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'beta')
    # Getting the type of 'oldb' (line 162)
    oldb_413698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'oldb')
    # Applying the binary operator 'div' (line 162)
    result_div_413699 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 21), 'div', beta_413697, oldb_413698)
    
    # Getting the type of 'r1' (line 162)
    r1_413700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'r1')
    # Applying the binary operator '*' (line 162)
    result_mul_413701 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 20), '*', result_div_413699, r1_413700)
    
    # Applying the binary operator '-' (line 162)
    result_sub_413702 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 16), '-', y_413696, result_mul_413701)
    
    # Assigning a type to the variable 'y' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'y', result_sub_413702)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to inner(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'v' (line 164)
    v_413704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'v', False)
    # Getting the type of 'y' (line 164)
    y_413705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'y', False)
    # Processing the call keyword arguments (line 164)
    kwargs_413706 = {}
    # Getting the type of 'inner' (line 164)
    inner_413703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'inner', False)
    # Calling inner(args, kwargs) (line 164)
    inner_call_result_413707 = invoke(stypy.reporting.localization.Localization(__file__, 164, 15), inner_413703, *[v_413704, y_413705], **kwargs_413706)
    
    # Assigning a type to the variable 'alfa' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'alfa', inner_call_result_413707)
    
    # Assigning a BinOp to a Name (line 165):
    
    # Assigning a BinOp to a Name (line 165):
    # Getting the type of 'y' (line 165)
    y_413708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'y')
    # Getting the type of 'alfa' (line 165)
    alfa_413709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'alfa')
    # Getting the type of 'beta' (line 165)
    beta_413710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'beta')
    # Applying the binary operator 'div' (line 165)
    result_div_413711 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 17), 'div', alfa_413709, beta_413710)
    
    # Getting the type of 'r2' (line 165)
    r2_413712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'r2')
    # Applying the binary operator '*' (line 165)
    result_mul_413713 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 16), '*', result_div_413711, r2_413712)
    
    # Applying the binary operator '-' (line 165)
    result_sub_413714 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 12), '-', y_413708, result_mul_413713)
    
    # Assigning a type to the variable 'y' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'y', result_sub_413714)
    
    # Assigning a Name to a Name (line 166):
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'r2' (line 166)
    r2_413715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 13), 'r2')
    # Assigning a type to the variable 'r1' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'r1', r2_413715)
    
    # Assigning a Name to a Name (line 167):
    
    # Assigning a Name to a Name (line 167):
    # Getting the type of 'y' (line 167)
    y_413716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 13), 'y')
    # Assigning a type to the variable 'r2' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'r2', y_413716)
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to psolve(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'r2' (line 168)
    r2_413718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'r2', False)
    # Processing the call keyword arguments (line 168)
    kwargs_413719 = {}
    # Getting the type of 'psolve' (line 168)
    psolve_413717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'psolve', False)
    # Calling psolve(args, kwargs) (line 168)
    psolve_call_result_413720 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), psolve_413717, *[r2_413718], **kwargs_413719)
    
    # Assigning a type to the variable 'y' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'y', psolve_call_result_413720)
    
    # Assigning a Name to a Name (line 169):
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'beta' (line 169)
    beta_413721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'beta')
    # Assigning a type to the variable 'oldb' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'oldb', beta_413721)
    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to inner(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'r2' (line 170)
    r2_413723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'r2', False)
    # Getting the type of 'y' (line 170)
    y_413724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'y', False)
    # Processing the call keyword arguments (line 170)
    kwargs_413725 = {}
    # Getting the type of 'inner' (line 170)
    inner_413722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'inner', False)
    # Calling inner(args, kwargs) (line 170)
    inner_call_result_413726 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), inner_413722, *[r2_413723, y_413724], **kwargs_413725)
    
    # Assigning a type to the variable 'beta' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'beta', inner_call_result_413726)
    
    
    # Getting the type of 'beta' (line 171)
    beta_413727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'beta')
    int_413728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 18), 'int')
    # Applying the binary operator '<' (line 171)
    result_lt_413729 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), '<', beta_413727, int_413728)
    
    # Testing the type of an if condition (line 171)
    if_condition_413730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 8), result_lt_413729)
    # Assigning a type to the variable 'if_condition_413730' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'if_condition_413730', if_condition_413730)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 172)
    # Processing the call arguments (line 172)
    str_413732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 29), 'str', 'non-symmetric matrix')
    # Processing the call keyword arguments (line 172)
    kwargs_413733 = {}
    # Getting the type of 'ValueError' (line 172)
    ValueError_413731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 172)
    ValueError_call_result_413734 = invoke(stypy.reporting.localization.Localization(__file__, 172, 18), ValueError_413731, *[str_413732], **kwargs_413733)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 172, 12), ValueError_call_result_413734, 'raise parameter', BaseException)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 173):
    
    # Assigning a Call to a Name (line 173):
    
    # Call to sqrt(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'beta' (line 173)
    beta_413736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'beta', False)
    # Processing the call keyword arguments (line 173)
    kwargs_413737 = {}
    # Getting the type of 'sqrt' (line 173)
    sqrt_413735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 173)
    sqrt_call_result_413738 = invoke(stypy.reporting.localization.Localization(__file__, 173, 15), sqrt_413735, *[beta_413736], **kwargs_413737)
    
    # Assigning a type to the variable 'beta' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'beta', sqrt_call_result_413738)
    
    # Getting the type of 'tnorm2' (line 174)
    tnorm2_413739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tnorm2')
    # Getting the type of 'alfa' (line 174)
    alfa_413740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'alfa')
    int_413741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 24), 'int')
    # Applying the binary operator '**' (line 174)
    result_pow_413742 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 18), '**', alfa_413740, int_413741)
    
    # Getting the type of 'oldb' (line 174)
    oldb_413743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 28), 'oldb')
    int_413744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'int')
    # Applying the binary operator '**' (line 174)
    result_pow_413745 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 28), '**', oldb_413743, int_413744)
    
    # Applying the binary operator '+' (line 174)
    result_add_413746 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 18), '+', result_pow_413742, result_pow_413745)
    
    # Getting the type of 'beta' (line 174)
    beta_413747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'beta')
    int_413748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 44), 'int')
    # Applying the binary operator '**' (line 174)
    result_pow_413749 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 38), '**', beta_413747, int_413748)
    
    # Applying the binary operator '+' (line 174)
    result_add_413750 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 36), '+', result_add_413746, result_pow_413749)
    
    # Applying the binary operator '+=' (line 174)
    result_iadd_413751 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 8), '+=', tnorm2_413739, result_add_413750)
    # Assigning a type to the variable 'tnorm2' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tnorm2', result_iadd_413751)
    
    
    
    # Getting the type of 'itn' (line 176)
    itn_413752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'itn')
    int_413753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'int')
    # Applying the binary operator '==' (line 176)
    result_eq_413754 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 11), '==', itn_413752, int_413753)
    
    # Testing the type of an if condition (line 176)
    if_condition_413755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 8), result_eq_413754)
    # Assigning a type to the variable 'if_condition_413755' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'if_condition_413755', if_condition_413755)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'beta' (line 177)
    beta_413756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'beta')
    # Getting the type of 'beta1' (line 177)
    beta1_413757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'beta1')
    # Applying the binary operator 'div' (line 177)
    result_div_413758 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), 'div', beta_413756, beta1_413757)
    
    int_413759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 29), 'int')
    # Getting the type of 'eps' (line 177)
    eps_413760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'eps')
    # Applying the binary operator '*' (line 177)
    result_mul_413761 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 29), '*', int_413759, eps_413760)
    
    # Applying the binary operator '<=' (line 177)
    result_le_413762 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), '<=', result_div_413758, result_mul_413761)
    
    # Testing the type of an if condition (line 177)
    if_condition_413763 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 12), result_le_413762)
    # Assigning a type to the variable 'if_condition_413763' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'if_condition_413763', if_condition_413763)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 178):
    
    # Assigning a Num to a Name (line 178):
    int_413764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 24), 'int')
    # Assigning a type to the variable 'istop' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'istop', int_413764)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to abs(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'alfa' (line 180)
    alfa_413766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'alfa', False)
    # Processing the call keyword arguments (line 180)
    kwargs_413767 = {}
    # Getting the type of 'abs' (line 180)
    abs_413765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 180)
    abs_call_result_413768 = invoke(stypy.reporting.localization.Localization(__file__, 180, 19), abs_413765, *[alfa_413766], **kwargs_413767)
    
    # Assigning a type to the variable 'gmax' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'gmax', abs_call_result_413768)
    
    # Assigning a Name to a Name (line 181):
    
    # Assigning a Name to a Name (line 181):
    # Getting the type of 'gmax' (line 181)
    gmax_413769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'gmax')
    # Assigning a type to the variable 'gmin' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'gmin', gmax_413769)
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 187):
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'epsln' (line 187)
    epsln_413770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 17), 'epsln')
    # Assigning a type to the variable 'oldeps' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'oldeps', epsln_413770)
    
    # Assigning a BinOp to a Name (line 188):
    
    # Assigning a BinOp to a Name (line 188):
    # Getting the type of 'cs' (line 188)
    cs_413771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'cs')
    # Getting the type of 'dbar' (line 188)
    dbar_413772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'dbar')
    # Applying the binary operator '*' (line 188)
    result_mul_413773 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 16), '*', cs_413771, dbar_413772)
    
    # Getting the type of 'sn' (line 188)
    sn_413774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'sn')
    # Getting the type of 'alfa' (line 188)
    alfa_413775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'alfa')
    # Applying the binary operator '*' (line 188)
    result_mul_413776 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 28), '*', sn_413774, alfa_413775)
    
    # Applying the binary operator '+' (line 188)
    result_add_413777 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 16), '+', result_mul_413773, result_mul_413776)
    
    # Assigning a type to the variable 'delta' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'delta', result_add_413777)
    
    # Assigning a BinOp to a Name (line 189):
    
    # Assigning a BinOp to a Name (line 189):
    # Getting the type of 'sn' (line 189)
    sn_413778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'sn')
    # Getting the type of 'dbar' (line 189)
    dbar_413779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'dbar')
    # Applying the binary operator '*' (line 189)
    result_mul_413780 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 15), '*', sn_413778, dbar_413779)
    
    # Getting the type of 'cs' (line 189)
    cs_413781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 27), 'cs')
    # Getting the type of 'alfa' (line 189)
    alfa_413782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'alfa')
    # Applying the binary operator '*' (line 189)
    result_mul_413783 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 27), '*', cs_413781, alfa_413782)
    
    # Applying the binary operator '-' (line 189)
    result_sub_413784 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 15), '-', result_mul_413780, result_mul_413783)
    
    # Assigning a type to the variable 'gbar' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'gbar', result_sub_413784)
    
    # Assigning a BinOp to a Name (line 190):
    
    # Assigning a BinOp to a Name (line 190):
    # Getting the type of 'sn' (line 190)
    sn_413785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'sn')
    # Getting the type of 'beta' (line 190)
    beta_413786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), 'beta')
    # Applying the binary operator '*' (line 190)
    result_mul_413787 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 16), '*', sn_413785, beta_413786)
    
    # Assigning a type to the variable 'epsln' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'epsln', result_mul_413787)
    
    # Assigning a BinOp to a Name (line 191):
    
    # Assigning a BinOp to a Name (line 191):
    
    # Getting the type of 'cs' (line 191)
    cs_413788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 17), 'cs')
    # Applying the 'usub' unary operator (line 191)
    result___neg___413789 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), 'usub', cs_413788)
    
    # Getting the type of 'beta' (line 191)
    beta_413790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'beta')
    # Applying the binary operator '*' (line 191)
    result_mul_413791 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), '*', result___neg___413789, beta_413790)
    
    # Assigning a type to the variable 'dbar' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'dbar', result_mul_413791)
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to norm(...): (line 192)
    # Processing the call arguments (line 192)
    
    # Obtaining an instance of the builtin type 'list' (line 192)
    list_413793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 192)
    # Adding element type (line 192)
    # Getting the type of 'gbar' (line 192)
    gbar_413794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'gbar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 20), list_413793, gbar_413794)
    # Adding element type (line 192)
    # Getting the type of 'dbar' (line 192)
    dbar_413795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'dbar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 20), list_413793, dbar_413795)
    
    # Processing the call keyword arguments (line 192)
    kwargs_413796 = {}
    # Getting the type of 'norm' (line 192)
    norm_413792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'norm', False)
    # Calling norm(args, kwargs) (line 192)
    norm_call_result_413797 = invoke(stypy.reporting.localization.Localization(__file__, 192, 15), norm_413792, *[list_413793], **kwargs_413796)
    
    # Assigning a type to the variable 'root' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'root', norm_call_result_413797)
    
    # Assigning a BinOp to a Name (line 193):
    
    # Assigning a BinOp to a Name (line 193):
    # Getting the type of 'phibar' (line 193)
    phibar_413798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'phibar')
    # Getting the type of 'root' (line 193)
    root_413799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'root')
    # Applying the binary operator '*' (line 193)
    result_mul_413800 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 17), '*', phibar_413798, root_413799)
    
    # Assigning a type to the variable 'Arnorm' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'Arnorm', result_mul_413800)
    
    # Assigning a Call to a Name (line 197):
    
    # Assigning a Call to a Name (line 197):
    
    # Call to norm(...): (line 197)
    # Processing the call arguments (line 197)
    
    # Obtaining an instance of the builtin type 'list' (line 197)
    list_413802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 197)
    # Adding element type (line 197)
    # Getting the type of 'gbar' (line 197)
    gbar_413803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 22), 'gbar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_413802, gbar_413803)
    # Adding element type (line 197)
    # Getting the type of 'beta' (line 197)
    beta_413804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 28), 'beta', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_413802, beta_413804)
    
    # Processing the call keyword arguments (line 197)
    kwargs_413805 = {}
    # Getting the type of 'norm' (line 197)
    norm_413801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'norm', False)
    # Calling norm(args, kwargs) (line 197)
    norm_call_result_413806 = invoke(stypy.reporting.localization.Localization(__file__, 197, 16), norm_413801, *[list_413802], **kwargs_413805)
    
    # Assigning a type to the variable 'gamma' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'gamma', norm_call_result_413806)
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to max(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'gamma' (line 198)
    gamma_413808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'gamma', False)
    # Getting the type of 'eps' (line 198)
    eps_413809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'eps', False)
    # Processing the call keyword arguments (line 198)
    kwargs_413810 = {}
    # Getting the type of 'max' (line 198)
    max_413807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'max', False)
    # Calling max(args, kwargs) (line 198)
    max_call_result_413811 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), max_413807, *[gamma_413808, eps_413809], **kwargs_413810)
    
    # Assigning a type to the variable 'gamma' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'gamma', max_call_result_413811)
    
    # Assigning a BinOp to a Name (line 199):
    
    # Assigning a BinOp to a Name (line 199):
    # Getting the type of 'gbar' (line 199)
    gbar_413812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'gbar')
    # Getting the type of 'gamma' (line 199)
    gamma_413813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'gamma')
    # Applying the binary operator 'div' (line 199)
    result_div_413814 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 13), 'div', gbar_413812, gamma_413813)
    
    # Assigning a type to the variable 'cs' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'cs', result_div_413814)
    
    # Assigning a BinOp to a Name (line 200):
    
    # Assigning a BinOp to a Name (line 200):
    # Getting the type of 'beta' (line 200)
    beta_413815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'beta')
    # Getting the type of 'gamma' (line 200)
    gamma_413816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'gamma')
    # Applying the binary operator 'div' (line 200)
    result_div_413817 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 13), 'div', beta_413815, gamma_413816)
    
    # Assigning a type to the variable 'sn' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'sn', result_div_413817)
    
    # Assigning a BinOp to a Name (line 201):
    
    # Assigning a BinOp to a Name (line 201):
    # Getting the type of 'cs' (line 201)
    cs_413818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 14), 'cs')
    # Getting the type of 'phibar' (line 201)
    phibar_413819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'phibar')
    # Applying the binary operator '*' (line 201)
    result_mul_413820 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 14), '*', cs_413818, phibar_413819)
    
    # Assigning a type to the variable 'phi' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'phi', result_mul_413820)
    
    # Assigning a BinOp to a Name (line 202):
    
    # Assigning a BinOp to a Name (line 202):
    # Getting the type of 'sn' (line 202)
    sn_413821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 17), 'sn')
    # Getting the type of 'phibar' (line 202)
    phibar_413822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'phibar')
    # Applying the binary operator '*' (line 202)
    result_mul_413823 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 17), '*', sn_413821, phibar_413822)
    
    # Assigning a type to the variable 'phibar' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'phibar', result_mul_413823)
    
    # Assigning a BinOp to a Name (line 206):
    
    # Assigning a BinOp to a Name (line 206):
    float_413824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 16), 'float')
    # Getting the type of 'gamma' (line 206)
    gamma_413825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'gamma')
    # Applying the binary operator 'div' (line 206)
    result_div_413826 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 16), 'div', float_413824, gamma_413825)
    
    # Assigning a type to the variable 'denom' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'denom', result_div_413826)
    
    # Assigning a Name to a Name (line 207):
    
    # Assigning a Name to a Name (line 207):
    # Getting the type of 'w2' (line 207)
    w2_413827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'w2')
    # Assigning a type to the variable 'w1' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'w1', w2_413827)
    
    # Assigning a Name to a Name (line 208):
    
    # Assigning a Name to a Name (line 208):
    # Getting the type of 'w' (line 208)
    w_413828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 13), 'w')
    # Assigning a type to the variable 'w2' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'w2', w_413828)
    
    # Assigning a BinOp to a Name (line 209):
    
    # Assigning a BinOp to a Name (line 209):
    # Getting the type of 'v' (line 209)
    v_413829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'v')
    # Getting the type of 'oldeps' (line 209)
    oldeps_413830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 17), 'oldeps')
    # Getting the type of 'w1' (line 209)
    w1_413831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'w1')
    # Applying the binary operator '*' (line 209)
    result_mul_413832 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 17), '*', oldeps_413830, w1_413831)
    
    # Applying the binary operator '-' (line 209)
    result_sub_413833 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 13), '-', v_413829, result_mul_413832)
    
    # Getting the type of 'delta' (line 209)
    delta_413834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 29), 'delta')
    # Getting the type of 'w2' (line 209)
    w2_413835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 35), 'w2')
    # Applying the binary operator '*' (line 209)
    result_mul_413836 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 29), '*', delta_413834, w2_413835)
    
    # Applying the binary operator '-' (line 209)
    result_sub_413837 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 27), '-', result_sub_413833, result_mul_413836)
    
    # Getting the type of 'denom' (line 209)
    denom_413838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 41), 'denom')
    # Applying the binary operator '*' (line 209)
    result_mul_413839 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 12), '*', result_sub_413837, denom_413838)
    
    # Assigning a type to the variable 'w' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'w', result_mul_413839)
    
    # Assigning a BinOp to a Name (line 210):
    
    # Assigning a BinOp to a Name (line 210):
    # Getting the type of 'x' (line 210)
    x_413840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'x')
    # Getting the type of 'phi' (line 210)
    phi_413841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'phi')
    # Getting the type of 'w' (line 210)
    w_413842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'w')
    # Applying the binary operator '*' (line 210)
    result_mul_413843 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 16), '*', phi_413841, w_413842)
    
    # Applying the binary operator '+' (line 210)
    result_add_413844 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 12), '+', x_413840, result_mul_413843)
    
    # Assigning a type to the variable 'x' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'x', result_add_413844)
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to max(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'gmax' (line 214)
    gmax_413846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'gmax', False)
    # Getting the type of 'gamma' (line 214)
    gamma_413847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 25), 'gamma', False)
    # Processing the call keyword arguments (line 214)
    kwargs_413848 = {}
    # Getting the type of 'max' (line 214)
    max_413845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'max', False)
    # Calling max(args, kwargs) (line 214)
    max_call_result_413849 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), max_413845, *[gmax_413846, gamma_413847], **kwargs_413848)
    
    # Assigning a type to the variable 'gmax' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'gmax', max_call_result_413849)
    
    # Assigning a Call to a Name (line 215):
    
    # Assigning a Call to a Name (line 215):
    
    # Call to min(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'gmin' (line 215)
    gmin_413851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'gmin', False)
    # Getting the type of 'gamma' (line 215)
    gamma_413852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 25), 'gamma', False)
    # Processing the call keyword arguments (line 215)
    kwargs_413853 = {}
    # Getting the type of 'min' (line 215)
    min_413850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'min', False)
    # Calling min(args, kwargs) (line 215)
    min_call_result_413854 = invoke(stypy.reporting.localization.Localization(__file__, 215, 15), min_413850, *[gmin_413851, gamma_413852], **kwargs_413853)
    
    # Assigning a type to the variable 'gmin' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'gmin', min_call_result_413854)
    
    # Assigning a BinOp to a Name (line 216):
    
    # Assigning a BinOp to a Name (line 216):
    # Getting the type of 'rhs1' (line 216)
    rhs1_413855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'rhs1')
    # Getting the type of 'gamma' (line 216)
    gamma_413856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'gamma')
    # Applying the binary operator 'div' (line 216)
    result_div_413857 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 12), 'div', rhs1_413855, gamma_413856)
    
    # Assigning a type to the variable 'z' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'z', result_div_413857)
    
    # Assigning a BinOp to a Name (line 217):
    
    # Assigning a BinOp to a Name (line 217):
    # Getting the type of 'z' (line 217)
    z_413858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), 'z')
    int_413859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 20), 'int')
    # Applying the binary operator '**' (line 217)
    result_pow_413860 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 17), '**', z_413858, int_413859)
    
    # Getting the type of 'ynorm2' (line 217)
    ynorm2_413861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'ynorm2')
    # Applying the binary operator '+' (line 217)
    result_add_413862 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 17), '+', result_pow_413860, ynorm2_413861)
    
    # Assigning a type to the variable 'ynorm2' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'ynorm2', result_add_413862)
    
    # Assigning a BinOp to a Name (line 218):
    
    # Assigning a BinOp to a Name (line 218):
    # Getting the type of 'rhs2' (line 218)
    rhs2_413863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'rhs2')
    # Getting the type of 'delta' (line 218)
    delta_413864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'delta')
    # Getting the type of 'z' (line 218)
    z_413865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'z')
    # Applying the binary operator '*' (line 218)
    result_mul_413866 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 22), '*', delta_413864, z_413865)
    
    # Applying the binary operator '-' (line 218)
    result_sub_413867 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 15), '-', rhs2_413863, result_mul_413866)
    
    # Assigning a type to the variable 'rhs1' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'rhs1', result_sub_413867)
    
    # Assigning a BinOp to a Name (line 219):
    
    # Assigning a BinOp to a Name (line 219):
    
    # Getting the type of 'epsln' (line 219)
    epsln_413868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 17), 'epsln')
    # Applying the 'usub' unary operator (line 219)
    result___neg___413869 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 15), 'usub', epsln_413868)
    
    # Getting the type of 'z' (line 219)
    z_413870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'z')
    # Applying the binary operator '*' (line 219)
    result_mul_413871 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 15), '*', result___neg___413869, z_413870)
    
    # Assigning a type to the variable 'rhs2' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'rhs2', result_mul_413871)
    
    # Assigning a Call to a Name (line 223):
    
    # Assigning a Call to a Name (line 223):
    
    # Call to sqrt(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'tnorm2' (line 223)
    tnorm2_413873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'tnorm2', False)
    # Processing the call keyword arguments (line 223)
    kwargs_413874 = {}
    # Getting the type of 'sqrt' (line 223)
    sqrt_413872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 223)
    sqrt_call_result_413875 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), sqrt_413872, *[tnorm2_413873], **kwargs_413874)
    
    # Assigning a type to the variable 'Anorm' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'Anorm', sqrt_call_result_413875)
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Call to sqrt(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'ynorm2' (line 224)
    ynorm2_413877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'ynorm2', False)
    # Processing the call keyword arguments (line 224)
    kwargs_413878 = {}
    # Getting the type of 'sqrt' (line 224)
    sqrt_413876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 224)
    sqrt_call_result_413879 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), sqrt_413876, *[ynorm2_413877], **kwargs_413878)
    
    # Assigning a type to the variable 'ynorm' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'ynorm', sqrt_call_result_413879)
    
    # Assigning a BinOp to a Name (line 225):
    
    # Assigning a BinOp to a Name (line 225):
    # Getting the type of 'Anorm' (line 225)
    Anorm_413880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'Anorm')
    # Getting the type of 'eps' (line 225)
    eps_413881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'eps')
    # Applying the binary operator '*' (line 225)
    result_mul_413882 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 15), '*', Anorm_413880, eps_413881)
    
    # Assigning a type to the variable 'epsa' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'epsa', result_mul_413882)
    
    # Assigning a BinOp to a Name (line 226):
    
    # Assigning a BinOp to a Name (line 226):
    # Getting the type of 'Anorm' (line 226)
    Anorm_413883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'Anorm')
    # Getting the type of 'ynorm' (line 226)
    ynorm_413884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'ynorm')
    # Applying the binary operator '*' (line 226)
    result_mul_413885 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 15), '*', Anorm_413883, ynorm_413884)
    
    # Getting the type of 'eps' (line 226)
    eps_413886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'eps')
    # Applying the binary operator '*' (line 226)
    result_mul_413887 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 29), '*', result_mul_413885, eps_413886)
    
    # Assigning a type to the variable 'epsx' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'epsx', result_mul_413887)
    
    # Assigning a BinOp to a Name (line 227):
    
    # Assigning a BinOp to a Name (line 227):
    # Getting the type of 'Anorm' (line 227)
    Anorm_413888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'Anorm')
    # Getting the type of 'ynorm' (line 227)
    ynorm_413889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'ynorm')
    # Applying the binary operator '*' (line 227)
    result_mul_413890 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 15), '*', Anorm_413888, ynorm_413889)
    
    # Getting the type of 'tol' (line 227)
    tol_413891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 31), 'tol')
    # Applying the binary operator '*' (line 227)
    result_mul_413892 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 29), '*', result_mul_413890, tol_413891)
    
    # Assigning a type to the variable 'epsr' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'epsr', result_mul_413892)
    
    # Assigning a Name to a Name (line 228):
    
    # Assigning a Name to a Name (line 228):
    # Getting the type of 'gbar' (line 228)
    gbar_413893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'gbar')
    # Assigning a type to the variable 'diag' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'diag', gbar_413893)
    
    
    # Getting the type of 'diag' (line 230)
    diag_413894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'diag')
    int_413895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 19), 'int')
    # Applying the binary operator '==' (line 230)
    result_eq_413896 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), '==', diag_413894, int_413895)
    
    # Testing the type of an if condition (line 230)
    if_condition_413897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_eq_413896)
    # Assigning a type to the variable 'if_condition_413897' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_413897', if_condition_413897)
    # SSA begins for if statement (line 230)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 231):
    
    # Assigning a Name to a Name (line 231):
    # Getting the type of 'epsa' (line 231)
    epsa_413898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'epsa')
    # Assigning a type to the variable 'diag' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'diag', epsa_413898)
    # SSA join for if statement (line 230)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 233):
    
    # Assigning a Name to a Name (line 233):
    # Getting the type of 'phibar' (line 233)
    phibar_413899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 17), 'phibar')
    # Assigning a type to the variable 'qrnorm' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'qrnorm', phibar_413899)
    
    # Assigning a Name to a Name (line 234):
    
    # Assigning a Name to a Name (line 234):
    # Getting the type of 'qrnorm' (line 234)
    qrnorm_413900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'qrnorm')
    # Assigning a type to the variable 'rnorm' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'rnorm', qrnorm_413900)
    
    # Assigning a BinOp to a Name (line 235):
    
    # Assigning a BinOp to a Name (line 235):
    # Getting the type of 'rnorm' (line 235)
    rnorm_413901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'rnorm')
    # Getting the type of 'Anorm' (line 235)
    Anorm_413902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'Anorm')
    # Getting the type of 'ynorm' (line 235)
    ynorm_413903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 31), 'ynorm')
    # Applying the binary operator '*' (line 235)
    result_mul_413904 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 25), '*', Anorm_413902, ynorm_413903)
    
    # Applying the binary operator 'div' (line 235)
    result_div_413905 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 16), 'div', rnorm_413901, result_mul_413904)
    
    # Assigning a type to the variable 'test1' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'test1', result_div_413905)
    
    # Assigning a BinOp to a Name (line 236):
    
    # Assigning a BinOp to a Name (line 236):
    # Getting the type of 'root' (line 236)
    root_413906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'root')
    # Getting the type of 'Anorm' (line 236)
    Anorm_413907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'Anorm')
    # Applying the binary operator 'div' (line 236)
    result_div_413908 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 16), 'div', root_413906, Anorm_413907)
    
    # Assigning a type to the variable 'test2' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'test2', result_div_413908)
    
    # Assigning a BinOp to a Name (line 244):
    
    # Assigning a BinOp to a Name (line 244):
    # Getting the type of 'gmax' (line 244)
    gmax_413909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'gmax')
    # Getting the type of 'gmin' (line 244)
    gmin_413910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 21), 'gmin')
    # Applying the binary operator 'div' (line 244)
    result_div_413911 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 16), 'div', gmax_413909, gmin_413910)
    
    # Assigning a type to the variable 'Acond' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'Acond', result_div_413911)
    
    
    # Getting the type of 'istop' (line 249)
    istop_413912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'istop')
    int_413913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'int')
    # Applying the binary operator '==' (line 249)
    result_eq_413914 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 11), '==', istop_413912, int_413913)
    
    # Testing the type of an if condition (line 249)
    if_condition_413915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), result_eq_413914)
    # Assigning a type to the variable 'if_condition_413915' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_413915', if_condition_413915)
    # SSA begins for if statement (line 249)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 250):
    
    # Assigning a BinOp to a Name (line 250):
    int_413916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 17), 'int')
    # Getting the type of 'test1' (line 250)
    test1_413917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 21), 'test1')
    # Applying the binary operator '+' (line 250)
    result_add_413918 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 17), '+', int_413916, test1_413917)
    
    # Assigning a type to the variable 't1' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 't1', result_add_413918)
    
    # Assigning a BinOp to a Name (line 251):
    
    # Assigning a BinOp to a Name (line 251):
    int_413919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'int')
    # Getting the type of 'test2' (line 251)
    test2_413920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 21), 'test2')
    # Applying the binary operator '+' (line 251)
    result_add_413921 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 17), '+', int_413919, test2_413920)
    
    # Assigning a type to the variable 't2' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 't2', result_add_413921)
    
    
    # Getting the type of 't2' (line 252)
    t2_413922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 't2')
    int_413923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 21), 'int')
    # Applying the binary operator '<=' (line 252)
    result_le_413924 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 15), '<=', t2_413922, int_413923)
    
    # Testing the type of an if condition (line 252)
    if_condition_413925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 12), result_le_413924)
    # Assigning a type to the variable 'if_condition_413925' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'if_condition_413925', if_condition_413925)
    # SSA begins for if statement (line 252)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 253):
    
    # Assigning a Num to a Name (line 253):
    int_413926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 24), 'int')
    # Assigning a type to the variable 'istop' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'istop', int_413926)
    # SSA join for if statement (line 252)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 't1' (line 254)
    t1_413927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 't1')
    int_413928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 21), 'int')
    # Applying the binary operator '<=' (line 254)
    result_le_413929 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 15), '<=', t1_413927, int_413928)
    
    # Testing the type of an if condition (line 254)
    if_condition_413930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 12), result_le_413929)
    # Assigning a type to the variable 'if_condition_413930' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'if_condition_413930', if_condition_413930)
    # SSA begins for if statement (line 254)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 255):
    
    # Assigning a Num to a Name (line 255):
    int_413931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 24), 'int')
    # Assigning a type to the variable 'istop' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'istop', int_413931)
    # SSA join for if statement (line 254)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 257)
    itn_413932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'itn')
    # Getting the type of 'maxiter' (line 257)
    maxiter_413933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'maxiter')
    # Applying the binary operator '>=' (line 257)
    result_ge_413934 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 15), '>=', itn_413932, maxiter_413933)
    
    # Testing the type of an if condition (line 257)
    if_condition_413935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 12), result_ge_413934)
    # Assigning a type to the variable 'if_condition_413935' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'if_condition_413935', if_condition_413935)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 258):
    
    # Assigning a Num to a Name (line 258):
    int_413936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 24), 'int')
    # Assigning a type to the variable 'istop' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'istop', int_413936)
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'Acond' (line 259)
    Acond_413937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'Acond')
    float_413938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 24), 'float')
    # Getting the type of 'eps' (line 259)
    eps_413939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'eps')
    # Applying the binary operator 'div' (line 259)
    result_div_413940 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 24), 'div', float_413938, eps_413939)
    
    # Applying the binary operator '>=' (line 259)
    result_ge_413941 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 15), '>=', Acond_413937, result_div_413940)
    
    # Testing the type of an if condition (line 259)
    if_condition_413942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 12), result_ge_413941)
    # Assigning a type to the variable 'if_condition_413942' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'if_condition_413942', if_condition_413942)
    # SSA begins for if statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 260):
    
    # Assigning a Num to a Name (line 260):
    int_413943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 24), 'int')
    # Assigning a type to the variable 'istop' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'istop', int_413943)
    # SSA join for if statement (line 259)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'epsx' (line 261)
    epsx_413944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'epsx')
    # Getting the type of 'beta' (line 261)
    beta_413945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'beta')
    # Applying the binary operator '>=' (line 261)
    result_ge_413946 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 15), '>=', epsx_413944, beta_413945)
    
    # Testing the type of an if condition (line 261)
    if_condition_413947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 12), result_ge_413946)
    # Assigning a type to the variable 'if_condition_413947' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'if_condition_413947', if_condition_413947)
    # SSA begins for if statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 262):
    
    # Assigning a Num to a Name (line 262):
    int_413948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 24), 'int')
    # Assigning a type to the variable 'istop' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'istop', int_413948)
    # SSA join for if statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test2' (line 265)
    test2_413949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'test2')
    # Getting the type of 'tol' (line 265)
    tol_413950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'tol')
    # Applying the binary operator '<=' (line 265)
    result_le_413951 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), '<=', test2_413949, tol_413950)
    
    # Testing the type of an if condition (line 265)
    if_condition_413952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 12), result_le_413951)
    # Assigning a type to the variable 'if_condition_413952' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'if_condition_413952', if_condition_413952)
    # SSA begins for if statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 266):
    
    # Assigning a Num to a Name (line 266):
    int_413953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 24), 'int')
    # Assigning a type to the variable 'istop' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'istop', int_413953)
    # SSA join for if statement (line 265)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test1' (line 267)
    test1_413954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'test1')
    # Getting the type of 'tol' (line 267)
    tol_413955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'tol')
    # Applying the binary operator '<=' (line 267)
    result_le_413956 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 15), '<=', test1_413954, tol_413955)
    
    # Testing the type of an if condition (line 267)
    if_condition_413957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 12), result_le_413956)
    # Assigning a type to the variable 'if_condition_413957' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'if_condition_413957', if_condition_413957)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 268):
    
    # Assigning a Num to a Name (line 268):
    int_413958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 24), 'int')
    # Assigning a type to the variable 'istop' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'istop', int_413958)
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 249)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 272):
    
    # Assigning a Name to a Name (line 272):
    # Getting the type of 'False' (line 272)
    False_413959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'False')
    # Assigning a type to the variable 'prnt' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'prnt', False_413959)
    
    
    # Getting the type of 'n' (line 273)
    n_413960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'n')
    int_413961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 16), 'int')
    # Applying the binary operator '<=' (line 273)
    result_le_413962 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), '<=', n_413960, int_413961)
    
    # Testing the type of an if condition (line 273)
    if_condition_413963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_le_413962)
    # Assigning a type to the variable 'if_condition_413963' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_413963', if_condition_413963)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 274):
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'True' (line 274)
    True_413964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'prnt', True_413964)
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 275)
    itn_413965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'itn')
    int_413966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 18), 'int')
    # Applying the binary operator '<=' (line 275)
    result_le_413967 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 11), '<=', itn_413965, int_413966)
    
    # Testing the type of an if condition (line 275)
    if_condition_413968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 8), result_le_413967)
    # Assigning a type to the variable 'if_condition_413968' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'if_condition_413968', if_condition_413968)
    # SSA begins for if statement (line 275)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 276):
    
    # Assigning a Name to a Name (line 276):
    # Getting the type of 'True' (line 276)
    True_413969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'prnt', True_413969)
    # SSA join for if statement (line 275)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 277)
    itn_413970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 11), 'itn')
    # Getting the type of 'maxiter' (line 277)
    maxiter_413971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'maxiter')
    int_413972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 26), 'int')
    # Applying the binary operator '-' (line 277)
    result_sub_413973 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 18), '-', maxiter_413971, int_413972)
    
    # Applying the binary operator '>=' (line 277)
    result_ge_413974 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 11), '>=', itn_413970, result_sub_413973)
    
    # Testing the type of an if condition (line 277)
    if_condition_413975 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 8), result_ge_413974)
    # Assigning a type to the variable 'if_condition_413975' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'if_condition_413975', if_condition_413975)
    # SSA begins for if statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 278):
    
    # Assigning a Name to a Name (line 278):
    # Getting the type of 'True' (line 278)
    True_413976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'prnt', True_413976)
    # SSA join for if statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 279)
    itn_413977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'itn')
    int_413978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 17), 'int')
    # Applying the binary operator '%' (line 279)
    result_mod_413979 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), '%', itn_413977, int_413978)
    
    int_413980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 23), 'int')
    # Applying the binary operator '==' (line 279)
    result_eq_413981 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), '==', result_mod_413979, int_413980)
    
    # Testing the type of an if condition (line 279)
    if_condition_413982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_eq_413981)
    # Assigning a type to the variable 'if_condition_413982' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_413982', if_condition_413982)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 280):
    
    # Assigning a Name to a Name (line 280):
    # Getting the type of 'True' (line 280)
    True_413983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'prnt', True_413983)
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'qrnorm' (line 281)
    qrnorm_413984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'qrnorm')
    int_413985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 21), 'int')
    # Getting the type of 'epsx' (line 281)
    epsx_413986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'epsx')
    # Applying the binary operator '*' (line 281)
    result_mul_413987 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 21), '*', int_413985, epsx_413986)
    
    # Applying the binary operator '<=' (line 281)
    result_le_413988 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 11), '<=', qrnorm_413984, result_mul_413987)
    
    # Testing the type of an if condition (line 281)
    if_condition_413989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 8), result_le_413988)
    # Assigning a type to the variable 'if_condition_413989' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'if_condition_413989', if_condition_413989)
    # SSA begins for if statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 282):
    
    # Assigning a Name to a Name (line 282):
    # Getting the type of 'True' (line 282)
    True_413990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'prnt', True_413990)
    # SSA join for if statement (line 281)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'qrnorm' (line 283)
    qrnorm_413991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'qrnorm')
    int_413992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 21), 'int')
    # Getting the type of 'epsr' (line 283)
    epsr_413993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'epsr')
    # Applying the binary operator '*' (line 283)
    result_mul_413994 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 21), '*', int_413992, epsr_413993)
    
    # Applying the binary operator '<=' (line 283)
    result_le_413995 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 11), '<=', qrnorm_413991, result_mul_413994)
    
    # Testing the type of an if condition (line 283)
    if_condition_413996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 8), result_le_413995)
    # Assigning a type to the variable 'if_condition_413996' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'if_condition_413996', if_condition_413996)
    # SSA begins for if statement (line 283)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 284):
    
    # Assigning a Name to a Name (line 284):
    # Getting the type of 'True' (line 284)
    True_413997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'prnt', True_413997)
    # SSA join for if statement (line 283)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'Acond' (line 285)
    Acond_413998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'Acond')
    float_413999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 20), 'float')
    # Getting the type of 'eps' (line 285)
    eps_414000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'eps')
    # Applying the binary operator 'div' (line 285)
    result_div_414001 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 20), 'div', float_413999, eps_414000)
    
    # Applying the binary operator '<=' (line 285)
    result_le_414002 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), '<=', Acond_413998, result_div_414001)
    
    # Testing the type of an if condition (line 285)
    if_condition_414003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_le_414002)
    # Assigning a type to the variable 'if_condition_414003' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_414003', if_condition_414003)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 286):
    
    # Assigning a Name to a Name (line 286):
    # Getting the type of 'True' (line 286)
    True_414004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'prnt', True_414004)
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'istop' (line 287)
    istop_414005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'istop')
    int_414006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 20), 'int')
    # Applying the binary operator '!=' (line 287)
    result_ne_414007 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 11), '!=', istop_414005, int_414006)
    
    # Testing the type of an if condition (line 287)
    if_condition_414008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), result_ne_414007)
    # Assigning a type to the variable 'if_condition_414008' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_414008', if_condition_414008)
    # SSA begins for if statement (line 287)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 288):
    
    # Assigning a Name to a Name (line 288):
    # Getting the type of 'True' (line 288)
    True_414009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'True')
    # Assigning a type to the variable 'prnt' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'prnt', True_414009)
    # SSA join for if statement (line 287)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'show' (line 290)
    show_414010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'show')
    # Getting the type of 'prnt' (line 290)
    prnt_414011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'prnt')
    # Applying the binary operator 'and' (line 290)
    result_and_keyword_414012 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), 'and', show_414010, prnt_414011)
    
    # Testing the type of an if condition (line 290)
    if_condition_414013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), result_and_keyword_414012)
    # Assigning a type to the variable 'if_condition_414013' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_414013', if_condition_414013)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 291):
    
    # Assigning a BinOp to a Name (line 291):
    str_414014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 19), 'str', '%6g %12.5e %10.3e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 291)
    tuple_414015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 291)
    # Adding element type (line 291)
    # Getting the type of 'itn' (line 291)
    itn_414016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 42), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 42), tuple_414015, itn_414016)
    # Adding element type (line 291)
    
    # Obtaining the type of the subscript
    int_414017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 49), 'int')
    # Getting the type of 'x' (line 291)
    x_414018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 47), 'x')
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___414019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 47), x_414018, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_414020 = invoke(stypy.reporting.localization.Localization(__file__, 291, 47), getitem___414019, int_414017)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 42), tuple_414015, subscript_call_result_414020)
    # Adding element type (line 291)
    # Getting the type of 'test1' (line 291)
    test1_414021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 53), 'test1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 42), tuple_414015, test1_414021)
    
    # Applying the binary operator '%' (line 291)
    result_mod_414022 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 19), '%', str_414014, tuple_414015)
    
    # Assigning a type to the variable 'str1' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'str1', result_mod_414022)
    
    # Assigning a BinOp to a Name (line 292):
    
    # Assigning a BinOp to a Name (line 292):
    str_414023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 19), 'str', ' %10.3e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 292)
    tuple_414024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 292)
    # Adding element type (line 292)
    # Getting the type of 'test2' (line 292)
    test2_414025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 32), 'test2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 32), tuple_414024, test2_414025)
    
    # Applying the binary operator '%' (line 292)
    result_mod_414026 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 19), '%', str_414023, tuple_414024)
    
    # Assigning a type to the variable 'str2' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'str2', result_mod_414026)
    
    # Assigning a BinOp to a Name (line 293):
    
    # Assigning a BinOp to a Name (line 293):
    str_414027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'str', ' %8.1e %8.1e %8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 293)
    tuple_414028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 293)
    # Adding element type (line 293)
    # Getting the type of 'Anorm' (line 293)
    Anorm_414029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 43), 'Anorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 43), tuple_414028, Anorm_414029)
    # Adding element type (line 293)
    # Getting the type of 'Acond' (line 293)
    Acond_414030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 50), 'Acond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 43), tuple_414028, Acond_414030)
    # Adding element type (line 293)
    # Getting the type of 'gbar' (line 293)
    gbar_414031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 57), 'gbar')
    # Getting the type of 'Anorm' (line 293)
    Anorm_414032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 62), 'Anorm')
    # Applying the binary operator 'div' (line 293)
    result_div_414033 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 57), 'div', gbar_414031, Anorm_414032)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 43), tuple_414028, result_div_414033)
    
    # Applying the binary operator '%' (line 293)
    result_mod_414034 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 19), '%', str_414027, tuple_414028)
    
    # Assigning a type to the variable 'str3' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'str3', result_mod_414034)
    
    # Call to print(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'str1' (line 295)
    str1_414036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'str1', False)
    # Getting the type of 'str2' (line 295)
    str2_414037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'str2', False)
    # Applying the binary operator '+' (line 295)
    result_add_414038 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 18), '+', str1_414036, str2_414037)
    
    # Getting the type of 'str3' (line 295)
    str3_414039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 32), 'str3', False)
    # Applying the binary operator '+' (line 295)
    result_add_414040 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 30), '+', result_add_414038, str3_414039)
    
    # Processing the call keyword arguments (line 295)
    kwargs_414041 = {}
    # Getting the type of 'print' (line 295)
    print_414035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'print', False)
    # Calling print(args, kwargs) (line 295)
    print_call_result_414042 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), print_414035, *[result_add_414040], **kwargs_414041)
    
    
    
    # Getting the type of 'itn' (line 297)
    itn_414043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), 'itn')
    int_414044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 21), 'int')
    # Applying the binary operator '%' (line 297)
    result_mod_414045 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 15), '%', itn_414043, int_414044)
    
    int_414046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 27), 'int')
    # Applying the binary operator '==' (line 297)
    result_eq_414047 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 15), '==', result_mod_414045, int_414046)
    
    # Testing the type of an if condition (line 297)
    if_condition_414048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 12), result_eq_414047)
    # Assigning a type to the variable 'if_condition_414048' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'if_condition_414048', if_condition_414048)
    # SSA begins for if statement (line 297)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 298)
    # Processing the call keyword arguments (line 298)
    kwargs_414050 = {}
    # Getting the type of 'print' (line 298)
    print_414049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'print', False)
    # Calling print(args, kwargs) (line 298)
    print_call_result_414051 = invoke(stypy.reporting.localization.Localization(__file__, 298, 16), print_414049, *[], **kwargs_414050)
    
    # SSA join for if statement (line 297)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 300)
    # Getting the type of 'callback' (line 300)
    callback_414052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'callback')
    # Getting the type of 'None' (line 300)
    None_414053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'None')
    
    (may_be_414054, more_types_in_union_414055) = may_not_be_none(callback_414052, None_414053)

    if may_be_414054:

        if more_types_in_union_414055:
            # Runtime conditional SSA (line 300)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'x' (line 301)
        x_414057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 'x', False)
        # Processing the call keyword arguments (line 301)
        kwargs_414058 = {}
        # Getting the type of 'callback' (line 301)
        callback_414056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'callback', False)
        # Calling callback(args, kwargs) (line 301)
        callback_call_result_414059 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), callback_414056, *[x_414057], **kwargs_414058)
        

        if more_types_in_union_414055:
            # SSA join for if statement (line 300)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'istop' (line 303)
    istop_414060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'istop')
    int_414061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 20), 'int')
    # Applying the binary operator '!=' (line 303)
    result_ne_414062 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 11), '!=', istop_414060, int_414061)
    
    # Testing the type of an if condition (line 303)
    if_condition_414063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 8), result_ne_414062)
    # Assigning a type to the variable 'if_condition_414063' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'if_condition_414063', if_condition_414063)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'show' (line 306)
    show_414064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 7), 'show')
    # Testing the type of an if condition (line 306)
    if_condition_414065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 4), show_414064)
    # Assigning a type to the variable 'if_condition_414065' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'if_condition_414065', if_condition_414065)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 307)
    # Processing the call keyword arguments (line 307)
    kwargs_414067 = {}
    # Getting the type of 'print' (line 307)
    print_414066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'print', False)
    # Calling print(args, kwargs) (line 307)
    print_call_result_414068 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), print_414066, *[], **kwargs_414067)
    
    
    # Call to print(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'last' (line 308)
    last_414070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 14), 'last', False)
    str_414071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 21), 'str', ' istop   =  %3g               itn   =%5g')
    
    # Obtaining an instance of the builtin type 'tuple' (line 308)
    tuple_414072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 308)
    # Adding element type (line 308)
    # Getting the type of 'istop' (line 308)
    istop_414073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 67), 'istop', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 67), tuple_414072, istop_414073)
    # Adding element type (line 308)
    # Getting the type of 'itn' (line 308)
    itn_414074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 73), 'itn', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 67), tuple_414072, itn_414074)
    
    # Applying the binary operator '%' (line 308)
    result_mod_414075 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 21), '%', str_414071, tuple_414072)
    
    # Applying the binary operator '+' (line 308)
    result_add_414076 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 14), '+', last_414070, result_mod_414075)
    
    # Processing the call keyword arguments (line 308)
    kwargs_414077 = {}
    # Getting the type of 'print' (line 308)
    print_414069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'print', False)
    # Calling print(args, kwargs) (line 308)
    print_call_result_414078 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), print_414069, *[result_add_414076], **kwargs_414077)
    
    
    # Call to print(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'last' (line 309)
    last_414080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 14), 'last', False)
    str_414081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 21), 'str', ' Anorm   =  %12.4e      Acond =  %12.4e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 309)
    tuple_414082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 309)
    # Adding element type (line 309)
    # Getting the type of 'Anorm' (line 309)
    Anorm_414083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 66), 'Anorm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 66), tuple_414082, Anorm_414083)
    # Adding element type (line 309)
    # Getting the type of 'Acond' (line 309)
    Acond_414084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 72), 'Acond', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 66), tuple_414082, Acond_414084)
    
    # Applying the binary operator '%' (line 309)
    result_mod_414085 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 21), '%', str_414081, tuple_414082)
    
    # Applying the binary operator '+' (line 309)
    result_add_414086 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 14), '+', last_414080, result_mod_414085)
    
    # Processing the call keyword arguments (line 309)
    kwargs_414087 = {}
    # Getting the type of 'print' (line 309)
    print_414079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'print', False)
    # Calling print(args, kwargs) (line 309)
    print_call_result_414088 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), print_414079, *[result_add_414086], **kwargs_414087)
    
    
    # Call to print(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'last' (line 310)
    last_414090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 14), 'last', False)
    str_414091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 21), 'str', ' rnorm   =  %12.4e      ynorm =  %12.4e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 310)
    tuple_414092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 310)
    # Adding element type (line 310)
    # Getting the type of 'rnorm' (line 310)
    rnorm_414093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 66), 'rnorm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 66), tuple_414092, rnorm_414093)
    # Adding element type (line 310)
    # Getting the type of 'ynorm' (line 310)
    ynorm_414094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 72), 'ynorm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 66), tuple_414092, ynorm_414094)
    
    # Applying the binary operator '%' (line 310)
    result_mod_414095 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 21), '%', str_414091, tuple_414092)
    
    # Applying the binary operator '+' (line 310)
    result_add_414096 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 14), '+', last_414090, result_mod_414095)
    
    # Processing the call keyword arguments (line 310)
    kwargs_414097 = {}
    # Getting the type of 'print' (line 310)
    print_414089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'print', False)
    # Calling print(args, kwargs) (line 310)
    print_call_result_414098 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), print_414089, *[result_add_414096], **kwargs_414097)
    
    
    # Call to print(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'last' (line 311)
    last_414100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 14), 'last', False)
    str_414101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 21), 'str', ' Arnorm  =  %12.4e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 311)
    tuple_414102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 311)
    # Adding element type (line 311)
    # Getting the type of 'Arnorm' (line 311)
    Arnorm_414103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 45), 'Arnorm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 45), tuple_414102, Arnorm_414103)
    
    # Applying the binary operator '%' (line 311)
    result_mod_414104 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 21), '%', str_414101, tuple_414102)
    
    # Applying the binary operator '+' (line 311)
    result_add_414105 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 14), '+', last_414100, result_mod_414104)
    
    # Processing the call keyword arguments (line 311)
    kwargs_414106 = {}
    # Getting the type of 'print' (line 311)
    print_414099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'print', False)
    # Calling print(args, kwargs) (line 311)
    print_call_result_414107 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), print_414099, *[result_add_414105], **kwargs_414106)
    
    
    # Call to print(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'last' (line 312)
    last_414109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 14), 'last', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'istop' (line 312)
    istop_414110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'istop', False)
    int_414111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 31), 'int')
    # Applying the binary operator '+' (line 312)
    result_add_414112 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 25), '+', istop_414110, int_414111)
    
    # Getting the type of 'msg' (line 312)
    msg_414113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 21), 'msg', False)
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___414114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 21), msg_414113, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_414115 = invoke(stypy.reporting.localization.Localization(__file__, 312, 21), getitem___414114, result_add_414112)
    
    # Applying the binary operator '+' (line 312)
    result_add_414116 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 14), '+', last_414109, subscript_call_result_414115)
    
    # Processing the call keyword arguments (line 312)
    kwargs_414117 = {}
    # Getting the type of 'print' (line 312)
    print_414108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'print', False)
    # Calling print(args, kwargs) (line 312)
    print_call_result_414118 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), print_414108, *[result_add_414116], **kwargs_414117)
    
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'istop' (line 314)
    istop_414119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 7), 'istop')
    int_414120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 16), 'int')
    # Applying the binary operator '==' (line 314)
    result_eq_414121 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 7), '==', istop_414119, int_414120)
    
    # Testing the type of an if condition (line 314)
    if_condition_414122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 4), result_eq_414121)
    # Assigning a type to the variable 'if_condition_414122' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'if_condition_414122', if_condition_414122)
    # SSA begins for if statement (line 314)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 315):
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'maxiter' (line 315)
    maxiter_414123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'maxiter')
    # Assigning a type to the variable 'info' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'info', maxiter_414123)
    # SSA branch for the else part of an if statement (line 314)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 317):
    
    # Assigning a Num to a Name (line 317):
    int_414124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 15), 'int')
    # Assigning a type to the variable 'info' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'info', int_414124)
    # SSA join for if statement (line 314)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 319)
    tuple_414125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 319)
    # Adding element type (line 319)
    
    # Call to postprocess(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'x' (line 319)
    x_414127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'x', False)
    # Processing the call keyword arguments (line 319)
    kwargs_414128 = {}
    # Getting the type of 'postprocess' (line 319)
    postprocess_414126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 319)
    postprocess_call_result_414129 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), postprocess_414126, *[x_414127], **kwargs_414128)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 12), tuple_414125, postprocess_call_result_414129)
    # Adding element type (line 319)
    # Getting the type of 'info' (line 319)
    info_414130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 27), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 12), tuple_414125, info_414130)
    
    # Assigning a type to the variable 'stypy_return_type' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type', tuple_414125)
    
    # ################# End of 'minres(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minres' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_414131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_414131)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minres'
    return stypy_return_type_414131

# Assigning a type to the variable 'minres' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'minres', minres)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 323, 4))
    
    # 'from scipy import ones, arange' statement (line 323)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    import_414132 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 323, 4), 'scipy')

    if (type(import_414132) is not StypyTypeError):

        if (import_414132 != 'pyd_module'):
            __import__(import_414132)
            sys_modules_414133 = sys.modules[import_414132]
            import_from_module(stypy.reporting.localization.Localization(__file__, 323, 4), 'scipy', sys_modules_414133.module_type_store, module_type_store, ['ones', 'arange'])
            nest_module(stypy.reporting.localization.Localization(__file__, 323, 4), __file__, sys_modules_414133, sys_modules_414133.module_type_store, module_type_store)
        else:
            from scipy import ones, arange

            import_from_module(stypy.reporting.localization.Localization(__file__, 323, 4), 'scipy', None, module_type_store, ['ones', 'arange'], [ones, arange])

    else:
        # Assigning a type to the variable 'scipy' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'scipy', import_414132)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 324, 4))
    
    # 'from scipy.linalg import norm' statement (line 324)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    import_414134 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 324, 4), 'scipy.linalg')

    if (type(import_414134) is not StypyTypeError):

        if (import_414134 != 'pyd_module'):
            __import__(import_414134)
            sys_modules_414135 = sys.modules[import_414134]
            import_from_module(stypy.reporting.localization.Localization(__file__, 324, 4), 'scipy.linalg', sys_modules_414135.module_type_store, module_type_store, ['norm'])
            nest_module(stypy.reporting.localization.Localization(__file__, 324, 4), __file__, sys_modules_414135, sys_modules_414135.module_type_store, module_type_store)
        else:
            from scipy.linalg import norm

            import_from_module(stypy.reporting.localization.Localization(__file__, 324, 4), 'scipy.linalg', None, module_type_store, ['norm'], [norm])

    else:
        # Assigning a type to the variable 'scipy.linalg' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'scipy.linalg', import_414134)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 325, 4))
    
    # 'from scipy.sparse import spdiags' statement (line 325)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    import_414136 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 325, 4), 'scipy.sparse')

    if (type(import_414136) is not StypyTypeError):

        if (import_414136 != 'pyd_module'):
            __import__(import_414136)
            sys_modules_414137 = sys.modules[import_414136]
            import_from_module(stypy.reporting.localization.Localization(__file__, 325, 4), 'scipy.sparse', sys_modules_414137.module_type_store, module_type_store, ['spdiags'])
            nest_module(stypy.reporting.localization.Localization(__file__, 325, 4), __file__, sys_modules_414137, sys_modules_414137.module_type_store, module_type_store)
        else:
            from scipy.sparse import spdiags

            import_from_module(stypy.reporting.localization.Localization(__file__, 325, 4), 'scipy.sparse', None, module_type_store, ['spdiags'], [spdiags])

    else:
        # Assigning a type to the variable 'scipy.sparse' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'scipy.sparse', import_414136)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    
    
    # Assigning a Num to a Name (line 327):
    
    # Assigning a Num to a Name (line 327):
    int_414138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'int')
    # Assigning a type to the variable 'n' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'n', int_414138)
    
    # Assigning a List to a Name (line 329):
    
    # Assigning a List to a Name (line 329):
    
    # Obtaining an instance of the builtin type 'list' (line 329)
    list_414139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 329)
    
    # Assigning a type to the variable 'residuals' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'residuals', list_414139)

    @norecursion
    def cb(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cb'
        module_type_store = module_type_store.open_function_context('cb', 331, 4, False)
        
        # Passed parameters checking function
        cb.stypy_localization = localization
        cb.stypy_type_of_self = None
        cb.stypy_type_store = module_type_store
        cb.stypy_function_name = 'cb'
        cb.stypy_param_names_list = ['x']
        cb.stypy_varargs_param_name = None
        cb.stypy_kwargs_param_name = None
        cb.stypy_call_defaults = defaults
        cb.stypy_call_varargs = varargs
        cb.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'cb', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cb', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cb(...)' code ##################

        
        # Call to append(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Call to norm(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'b' (line 332)
        b_414143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 30), 'b', False)
        # Getting the type of 'A' (line 332)
        A_414144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'A', False)
        # Getting the type of 'x' (line 332)
        x_414145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 36), 'x', False)
        # Applying the binary operator '*' (line 332)
        result_mul_414146 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 34), '*', A_414144, x_414145)
        
        # Applying the binary operator '-' (line 332)
        result_sub_414147 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 30), '-', b_414143, result_mul_414146)
        
        # Processing the call keyword arguments (line 332)
        kwargs_414148 = {}
        # Getting the type of 'norm' (line 332)
        norm_414142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'norm', False)
        # Calling norm(args, kwargs) (line 332)
        norm_call_result_414149 = invoke(stypy.reporting.localization.Localization(__file__, 332, 25), norm_414142, *[result_sub_414147], **kwargs_414148)
        
        # Processing the call keyword arguments (line 332)
        kwargs_414150 = {}
        # Getting the type of 'residuals' (line 332)
        residuals_414140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'residuals', False)
        # Obtaining the member 'append' of a type (line 332)
        append_414141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), residuals_414140, 'append')
        # Calling append(args, kwargs) (line 332)
        append_call_result_414151 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), append_414141, *[norm_call_result_414149], **kwargs_414150)
        
        
        # ################# End of 'cb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cb' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_414152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_414152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cb'
        return stypy_return_type_414152

    # Assigning a type to the variable 'cb' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'cb', cb)
    
    # Assigning a Call to a Name (line 335):
    
    # Assigning a Call to a Name (line 335):
    
    # Call to spdiags(...): (line 335)
    # Processing the call arguments (line 335)
    
    # Obtaining an instance of the builtin type 'list' (line 335)
    list_414154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 335)
    # Adding element type (line 335)
    
    # Call to arange(...): (line 335)
    # Processing the call arguments (line 335)
    int_414156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 24), 'int')
    # Getting the type of 'n' (line 335)
    n_414157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 26), 'n', False)
    int_414158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 28), 'int')
    # Applying the binary operator '+' (line 335)
    result_add_414159 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 26), '+', n_414157, int_414158)
    
    # Processing the call keyword arguments (line 335)
    # Getting the type of 'float' (line 335)
    float_414160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 36), 'float', False)
    keyword_414161 = float_414160
    kwargs_414162 = {'dtype': keyword_414161}
    # Getting the type of 'arange' (line 335)
    arange_414155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'arange', False)
    # Calling arange(args, kwargs) (line 335)
    arange_call_result_414163 = invoke(stypy.reporting.localization.Localization(__file__, 335, 17), arange_414155, *[int_414156, result_add_414159], **kwargs_414162)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 16), list_414154, arange_call_result_414163)
    
    
    # Obtaining an instance of the builtin type 'list' (line 335)
    list_414164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 335)
    # Adding element type (line 335)
    int_414165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 45), list_414164, int_414165)
    
    # Getting the type of 'n' (line 335)
    n_414166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 50), 'n', False)
    # Getting the type of 'n' (line 335)
    n_414167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 53), 'n', False)
    # Processing the call keyword arguments (line 335)
    str_414168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 63), 'str', 'csr')
    keyword_414169 = str_414168
    kwargs_414170 = {'format': keyword_414169}
    # Getting the type of 'spdiags' (line 335)
    spdiags_414153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'spdiags', False)
    # Calling spdiags(args, kwargs) (line 335)
    spdiags_call_result_414171 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), spdiags_414153, *[list_414154, list_414164, n_414166, n_414167], **kwargs_414170)
    
    # Assigning a type to the variable 'A' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'A', spdiags_call_result_414171)
    
    # Assigning a Call to a Name (line 336):
    
    # Assigning a Call to a Name (line 336):
    
    # Call to spdiags(...): (line 336)
    # Processing the call arguments (line 336)
    
    # Obtaining an instance of the builtin type 'list' (line 336)
    list_414173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 336)
    # Adding element type (line 336)
    float_414174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 17), 'float')
    
    # Call to arange(...): (line 336)
    # Processing the call arguments (line 336)
    int_414176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 28), 'int')
    # Getting the type of 'n' (line 336)
    n_414177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), 'n', False)
    int_414178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 32), 'int')
    # Applying the binary operator '+' (line 336)
    result_add_414179 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 30), '+', n_414177, int_414178)
    
    # Processing the call keyword arguments (line 336)
    # Getting the type of 'float' (line 336)
    float_414180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'float', False)
    keyword_414181 = float_414180
    kwargs_414182 = {'dtype': keyword_414181}
    # Getting the type of 'arange' (line 336)
    arange_414175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'arange', False)
    # Calling arange(args, kwargs) (line 336)
    arange_call_result_414183 = invoke(stypy.reporting.localization.Localization(__file__, 336, 21), arange_414175, *[int_414176, result_add_414179], **kwargs_414182)
    
    # Applying the binary operator 'div' (line 336)
    result_div_414184 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 17), 'div', float_414174, arange_call_result_414183)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 16), list_414173, result_div_414184)
    
    
    # Obtaining an instance of the builtin type 'list' (line 336)
    list_414185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 336)
    # Adding element type (line 336)
    int_414186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 49), list_414185, int_414186)
    
    # Getting the type of 'n' (line 336)
    n_414187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 54), 'n', False)
    # Getting the type of 'n' (line 336)
    n_414188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 57), 'n', False)
    # Processing the call keyword arguments (line 336)
    str_414189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 67), 'str', 'csr')
    keyword_414190 = str_414189
    kwargs_414191 = {'format': keyword_414190}
    # Getting the type of 'spdiags' (line 336)
    spdiags_414172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'spdiags', False)
    # Calling spdiags(args, kwargs) (line 336)
    spdiags_call_result_414192 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), spdiags_414172, *[list_414173, list_414185, n_414187, n_414188], **kwargs_414191)
    
    # Assigning a type to the variable 'M' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'M', spdiags_call_result_414192)
    
    # Assigning a Attribute to a Attribute (line 337):
    
    # Assigning a Attribute to a Attribute (line 337):
    # Getting the type of 'M' (line 337)
    M_414193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'M')
    # Obtaining the member 'matvec' of a type (line 337)
    matvec_414194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 15), M_414193, 'matvec')
    # Getting the type of 'A' (line 337)
    A_414195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'A')
    # Setting the type of the member 'psolve' of a type (line 337)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 4), A_414195, 'psolve', matvec_414194)
    
    # Assigning a BinOp to a Name (line 338):
    
    # Assigning a BinOp to a Name (line 338):
    int_414196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 8), 'int')
    
    # Call to ones(...): (line 338)
    # Processing the call arguments (line 338)
    
    # Obtaining the type of the subscript
    int_414198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 23), 'int')
    # Getting the type of 'A' (line 338)
    A_414199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'A', False)
    # Obtaining the member 'shape' of a type (line 338)
    shape_414200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 15), A_414199, 'shape')
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___414201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 15), shape_414200, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_414202 = invoke(stypy.reporting.localization.Localization(__file__, 338, 15), getitem___414201, int_414198)
    
    # Processing the call keyword arguments (line 338)
    kwargs_414203 = {}
    # Getting the type of 'ones' (line 338)
    ones_414197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 10), 'ones', False)
    # Calling ones(args, kwargs) (line 338)
    ones_call_result_414204 = invoke(stypy.reporting.localization.Localization(__file__, 338, 10), ones_414197, *[subscript_call_result_414202], **kwargs_414203)
    
    # Applying the binary operator '*' (line 338)
    result_mul_414205 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 8), '*', int_414196, ones_call_result_414204)
    
    # Assigning a type to the variable 'b' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'b', result_mul_414205)
    
    # Assigning a Call to a Name (line 339):
    
    # Assigning a Call to a Name (line 339):
    
    # Call to minres(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'A' (line 339)
    A_414207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'A', False)
    # Getting the type of 'b' (line 339)
    b_414208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'b', False)
    # Processing the call keyword arguments (line 339)
    float_414209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 23), 'float')
    keyword_414210 = float_414209
    # Getting the type of 'None' (line 339)
    None_414211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 37), 'None', False)
    keyword_414212 = None_414211
    # Getting the type of 'cb' (line 339)
    cb_414213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 51), 'cb', False)
    keyword_414214 = cb_414213
    kwargs_414215 = {'callback': keyword_414214, 'tol': keyword_414210, 'maxiter': keyword_414212}
    # Getting the type of 'minres' (line 339)
    minres_414206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'minres', False)
    # Calling minres(args, kwargs) (line 339)
    minres_call_result_414216 = invoke(stypy.reporting.localization.Localization(__file__, 339, 8), minres_414206, *[A_414207, b_414208], **kwargs_414215)
    
    # Assigning a type to the variable 'x' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'x', minres_call_result_414216)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
