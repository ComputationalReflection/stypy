
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Utililty classes and functions for the polynomial modules.
3: 
4: This module provides: error and warning objects; a polynomial base class;
5: and some routines used in both the `polynomial` and `chebyshev` modules.
6: 
7: Error objects
8: -------------
9: 
10: .. autosummary::
11:    :toctree: generated/
12: 
13:    PolyError            base class for this sub-package's errors.
14:    PolyDomainError      raised when domains are mismatched.
15: 
16: Warning objects
17: ---------------
18: 
19: .. autosummary::
20:    :toctree: generated/
21: 
22:    RankWarning  raised in least-squares fit for rank-deficient matrix.
23: 
24: Base class
25: ----------
26: 
27: .. autosummary::
28:    :toctree: generated/
29: 
30:    PolyBase Obsolete base class for the polynomial classes. Do not use.
31: 
32: Functions
33: ---------
34: 
35: .. autosummary::
36:    :toctree: generated/
37: 
38:    as_series    convert list of array_likes into 1-D arrays of common type.
39:    trimseq      remove trailing zeros.
40:    trimcoef     remove small trailing coefficients.
41:    getdomain    return the domain appropriate for a given set of abscissae.
42:    mapdomain    maps points between domains.
43:    mapparms     parameters of the linear map between domains.
44: 
45: '''
46: from __future__ import division, absolute_import, print_function
47: 
48: import numpy as np
49: 
50: __all__ = [
51:     'RankWarning', 'PolyError', 'PolyDomainError', 'as_series', 'trimseq',
52:     'trimcoef', 'getdomain', 'mapdomain', 'mapparms', 'PolyBase']
53: 
54: #
55: # Warnings and Exceptions
56: #
57: 
58: class RankWarning(UserWarning):
59:     '''Issued by chebfit when the design matrix is rank deficient.'''
60:     pass
61: 
62: class PolyError(Exception):
63:     '''Base class for errors in this module.'''
64:     pass
65: 
66: class PolyDomainError(PolyError):
67:     '''Issued by the generic Poly class when two domains don't match.
68: 
69:     This is raised when an binary operation is passed Poly objects with
70:     different domains.
71: 
72:     '''
73:     pass
74: 
75: #
76: # Base class for all polynomial types
77: #
78: 
79: class PolyBase(object):
80:     '''
81:     Base class for all polynomial types.
82: 
83:     Deprecated in numpy 1.9.0, use the abstract
84:     ABCPolyBase class instead. Note that the latter
85:     reguires a number of virtual functions to be
86:     implemented.
87: 
88:     '''
89:     pass
90: 
91: #
92: # Helper functions to convert inputs to 1-D arrays
93: #
94: def trimseq(seq):
95:     '''Remove small Poly series coefficients.
96: 
97:     Parameters
98:     ----------
99:     seq : sequence
100:         Sequence of Poly series coefficients. This routine fails for
101:         empty sequences.
102: 
103:     Returns
104:     -------
105:     series : sequence
106:         Subsequence with trailing zeros removed. If the resulting sequence
107:         would be empty, return the first element. The returned sequence may
108:         or may not be a view.
109: 
110:     Notes
111:     -----
112:     Do not lose the type info if the sequence contains unknown objects.
113: 
114:     '''
115:     if len(seq) == 0:
116:         return seq
117:     else:
118:         for i in range(len(seq) - 1, -1, -1):
119:             if seq[i] != 0:
120:                 break
121:         return seq[:i+1]
122: 
123: 
124: def as_series(alist, trim=True):
125:     '''
126:     Return argument as a list of 1-d arrays.
127: 
128:     The returned list contains array(s) of dtype double, complex double, or
129:     object.  A 1-d argument of shape ``(N,)`` is parsed into ``N`` arrays of
130:     size one; a 2-d argument of shape ``(M,N)`` is parsed into ``M`` arrays
131:     of size ``N`` (i.e., is "parsed by row"); and a higher dimensional array
132:     raises a Value Error if it is not first reshaped into either a 1-d or 2-d
133:     array.
134: 
135:     Parameters
136:     ----------
137:     alist : array_like
138:         A 1- or 2-d array_like
139:     trim : boolean, optional
140:         When True, trailing zeros are removed from the inputs.
141:         When False, the inputs are passed through intact.
142: 
143:     Returns
144:     -------
145:     [a1, a2,...] : list of 1-D arrays
146:         A copy of the input data as a list of 1-d arrays.
147: 
148:     Raises
149:     ------
150:     ValueError
151:         Raised when `as_series` cannot convert its input to 1-d arrays, or at
152:         least one of the resulting arrays is empty.
153: 
154:     Examples
155:     --------
156:     >>> from numpy import polynomial as P
157:     >>> a = np.arange(4)
158:     >>> P.as_series(a)
159:     [array([ 0.]), array([ 1.]), array([ 2.]), array([ 3.])]
160:     >>> b = np.arange(6).reshape((2,3))
161:     >>> P.as_series(b)
162:     [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.])]
163: 
164:     '''
165:     arrays = [np.array(a, ndmin=1, copy=0) for a in alist]
166:     if min([a.size for a in arrays]) == 0:
167:         raise ValueError("Coefficient array is empty")
168:     if any([a.ndim != 1 for a in arrays]):
169:         raise ValueError("Coefficient array is not 1-d")
170:     if trim:
171:         arrays = [trimseq(a) for a in arrays]
172: 
173:     if any([a.dtype == np.dtype(object) for a in arrays]):
174:         ret = []
175:         for a in arrays:
176:             if a.dtype != np.dtype(object):
177:                 tmp = np.empty(len(a), dtype=np.dtype(object))
178:                 tmp[:] = a[:]
179:                 ret.append(tmp)
180:             else:
181:                 ret.append(a.copy())
182:     else:
183:         try:
184:             dtype = np.common_type(*arrays)
185:         except:
186:             raise ValueError("Coefficient arrays have no common type")
187:         ret = [np.array(a, copy=1, dtype=dtype) for a in arrays]
188:     return ret
189: 
190: 
191: def trimcoef(c, tol=0):
192:     '''
193:     Remove "small" "trailing" coefficients from a polynomial.
194: 
195:     "Small" means "small in absolute value" and is controlled by the
196:     parameter `tol`; "trailing" means highest order coefficient(s), e.g., in
197:     ``[0, 1, 1, 0, 0]`` (which represents ``0 + x + x**2 + 0*x**3 + 0*x**4``)
198:     both the 3-rd and 4-th order coefficients would be "trimmed."
199: 
200:     Parameters
201:     ----------
202:     c : array_like
203:         1-d array of coefficients, ordered from lowest order to highest.
204:     tol : number, optional
205:         Trailing (i.e., highest order) elements with absolute value less
206:         than or equal to `tol` (default value is zero) are removed.
207: 
208:     Returns
209:     -------
210:     trimmed : ndarray
211:         1-d array with trailing zeros removed.  If the resulting series
212:         would be empty, a series containing a single zero is returned.
213: 
214:     Raises
215:     ------
216:     ValueError
217:         If `tol` < 0
218: 
219:     See Also
220:     --------
221:     trimseq
222: 
223:     Examples
224:     --------
225:     >>> from numpy import polynomial as P
226:     >>> P.trimcoef((0,0,3,0,5,0,0))
227:     array([ 0.,  0.,  3.,  0.,  5.])
228:     >>> P.trimcoef((0,0,1e-3,0,1e-5,0,0),1e-3) # item == tol is trimmed
229:     array([ 0.])
230:     >>> i = complex(0,1) # works for complex
231:     >>> P.trimcoef((3e-4,1e-3*(1-i),5e-4,2e-5*(1+i)), 1e-3)
232:     array([ 0.0003+0.j   ,  0.0010-0.001j])
233: 
234:     '''
235:     if tol < 0:
236:         raise ValueError("tol must be non-negative")
237: 
238:     [c] = as_series([c])
239:     [ind] = np.where(np.abs(c) > tol)
240:     if len(ind) == 0:
241:         return c[:1]*0
242:     else:
243:         return c[:ind[-1] + 1].copy()
244: 
245: def getdomain(x):
246:     '''
247:     Return a domain suitable for given abscissae.
248: 
249:     Find a domain suitable for a polynomial or Chebyshev series
250:     defined at the values supplied.
251: 
252:     Parameters
253:     ----------
254:     x : array_like
255:         1-d array of abscissae whose domain will be determined.
256: 
257:     Returns
258:     -------
259:     domain : ndarray
260:         1-d array containing two values.  If the inputs are complex, then
261:         the two returned points are the lower left and upper right corners
262:         of the smallest rectangle (aligned with the axes) in the complex
263:         plane containing the points `x`. If the inputs are real, then the
264:         two points are the ends of the smallest interval containing the
265:         points `x`.
266: 
267:     See Also
268:     --------
269:     mapparms, mapdomain
270: 
271:     Examples
272:     --------
273:     >>> from numpy.polynomial import polyutils as pu
274:     >>> points = np.arange(4)**2 - 5; points
275:     array([-5, -4, -1,  4])
276:     >>> pu.getdomain(points)
277:     array([-5.,  4.])
278:     >>> c = np.exp(complex(0,1)*np.pi*np.arange(12)/6) # unit circle
279:     >>> pu.getdomain(c)
280:     array([-1.-1.j,  1.+1.j])
281: 
282:     '''
283:     [x] = as_series([x], trim=False)
284:     if x.dtype.char in np.typecodes['Complex']:
285:         rmin, rmax = x.real.min(), x.real.max()
286:         imin, imax = x.imag.min(), x.imag.max()
287:         return np.array((complex(rmin, imin), complex(rmax, imax)))
288:     else:
289:         return np.array((x.min(), x.max()))
290: 
291: def mapparms(old, new):
292:     '''
293:     Linear map parameters between domains.
294: 
295:     Return the parameters of the linear map ``offset + scale*x`` that maps
296:     `old` to `new` such that ``old[i] -> new[i]``, ``i = 0, 1``.
297: 
298:     Parameters
299:     ----------
300:     old, new : array_like
301:         Domains. Each domain must (successfully) convert to a 1-d array
302:         containing precisely two values.
303: 
304:     Returns
305:     -------
306:     offset, scale : scalars
307:         The map ``L(x) = offset + scale*x`` maps the first domain to the
308:         second.
309: 
310:     See Also
311:     --------
312:     getdomain, mapdomain
313: 
314:     Notes
315:     -----
316:     Also works for complex numbers, and thus can be used to calculate the
317:     parameters required to map any line in the complex plane to any other
318:     line therein.
319: 
320:     Examples
321:     --------
322:     >>> from numpy import polynomial as P
323:     >>> P.mapparms((-1,1),(-1,1))
324:     (0.0, 1.0)
325:     >>> P.mapparms((1,-1),(-1,1))
326:     (0.0, -1.0)
327:     >>> i = complex(0,1)
328:     >>> P.mapparms((-i,-1),(1,i))
329:     ((1+1j), (1+0j))
330: 
331:     '''
332:     oldlen = old[1] - old[0]
333:     newlen = new[1] - new[0]
334:     off = (old[1]*new[0] - old[0]*new[1])/oldlen
335:     scl = newlen/oldlen
336:     return off, scl
337: 
338: def mapdomain(x, old, new):
339:     '''
340:     Apply linear map to input points.
341: 
342:     The linear map ``offset + scale*x`` that maps the domain `old` to
343:     the domain `new` is applied to the points `x`.
344: 
345:     Parameters
346:     ----------
347:     x : array_like
348:         Points to be mapped. If `x` is a subtype of ndarray the subtype
349:         will be preserved.
350:     old, new : array_like
351:         The two domains that determine the map.  Each must (successfully)
352:         convert to 1-d arrays containing precisely two values.
353: 
354:     Returns
355:     -------
356:     x_out : ndarray
357:         Array of points of the same shape as `x`, after application of the
358:         linear map between the two domains.
359: 
360:     See Also
361:     --------
362:     getdomain, mapparms
363: 
364:     Notes
365:     -----
366:     Effectively, this implements:
367: 
368:     .. math ::
369:         x\\_out = new[0] + m(x - old[0])
370: 
371:     where
372: 
373:     .. math ::
374:         m = \\frac{new[1]-new[0]}{old[1]-old[0]}
375: 
376:     Examples
377:     --------
378:     >>> from numpy import polynomial as P
379:     >>> old_domain = (-1,1)
380:     >>> new_domain = (0,2*np.pi)
381:     >>> x = np.linspace(-1,1,6); x
382:     array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ])
383:     >>> x_out = P.mapdomain(x, old_domain, new_domain); x_out
384:     array([ 0.        ,  1.25663706,  2.51327412,  3.76991118,  5.02654825,
385:             6.28318531])
386:     >>> x - P.mapdomain(x_out, new_domain, old_domain)
387:     array([ 0.,  0.,  0.,  0.,  0.,  0.])
388: 
389:     Also works for complex numbers (and thus can be used to map any line in
390:     the complex plane to any other line therein).
391: 
392:     >>> i = complex(0,1)
393:     >>> old = (-1 - i, 1 + i)
394:     >>> new = (-1 + i, 1 - i)
395:     >>> z = np.linspace(old[0], old[1], 6); z
396:     array([-1.0-1.j , -0.6-0.6j, -0.2-0.2j,  0.2+0.2j,  0.6+0.6j,  1.0+1.j ])
397:     >>> new_z = P.mapdomain(z, old, new); new_z
398:     array([-1.0+1.j , -0.6+0.6j, -0.2+0.2j,  0.2-0.2j,  0.6-0.6j,  1.0-1.j ])
399: 
400:     '''
401:     x = np.asanyarray(x)
402:     off, scl = mapparms(old, new)
403:     return off + scl*x
404: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_178746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', "\nUtililty classes and functions for the polynomial modules.\n\nThis module provides: error and warning objects; a polynomial base class;\nand some routines used in both the `polynomial` and `chebyshev` modules.\n\nError objects\n-------------\n\n.. autosummary::\n   :toctree: generated/\n\n   PolyError            base class for this sub-package's errors.\n   PolyDomainError      raised when domains are mismatched.\n\nWarning objects\n---------------\n\n.. autosummary::\n   :toctree: generated/\n\n   RankWarning  raised in least-squares fit for rank-deficient matrix.\n\nBase class\n----------\n\n.. autosummary::\n   :toctree: generated/\n\n   PolyBase Obsolete base class for the polynomial classes. Do not use.\n\nFunctions\n---------\n\n.. autosummary::\n   :toctree: generated/\n\n   as_series    convert list of array_likes into 1-D arrays of common type.\n   trimseq      remove trailing zeros.\n   trimcoef     remove small trailing coefficients.\n   getdomain    return the domain appropriate for a given set of abscissae.\n   mapdomain    maps points between domains.\n   mapparms     parameters of the linear map between domains.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'import numpy' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_178747 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy')

if (type(import_178747) is not StypyTypeError):

    if (import_178747 != 'pyd_module'):
        __import__(import_178747)
        sys_modules_178748 = sys.modules[import_178747]
        import_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'np', sys_modules_178748.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy', import_178747)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a List to a Name (line 50):

# Assigning a List to a Name (line 50):
__all__ = ['RankWarning', 'PolyError', 'PolyDomainError', 'as_series', 'trimseq', 'trimcoef', 'getdomain', 'mapdomain', 'mapparms', 'PolyBase']
module_type_store.set_exportable_members(['RankWarning', 'PolyError', 'PolyDomainError', 'as_series', 'trimseq', 'trimcoef', 'getdomain', 'mapdomain', 'mapparms', 'PolyBase'])

# Obtaining an instance of the builtin type 'list' (line 50)
list_178749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 50)
# Adding element type (line 50)
str_178750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'str', 'RankWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178750)
# Adding element type (line 50)
str_178751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'str', 'PolyError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178751)
# Adding element type (line 50)
str_178752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 32), 'str', 'PolyDomainError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178752)
# Adding element type (line 50)
str_178753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 51), 'str', 'as_series')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178753)
# Adding element type (line 50)
str_178754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 64), 'str', 'trimseq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178754)
# Adding element type (line 50)
str_178755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'str', 'trimcoef')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178755)
# Adding element type (line 50)
str_178756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 16), 'str', 'getdomain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178756)
# Adding element type (line 50)
str_178757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'str', 'mapdomain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178757)
# Adding element type (line 50)
str_178758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 42), 'str', 'mapparms')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178758)
# Adding element type (line 50)
str_178759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 54), 'str', 'PolyBase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_178749, str_178759)

# Assigning a type to the variable '__all__' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '__all__', list_178749)
# Declaration of the 'RankWarning' class
# Getting the type of 'UserWarning' (line 58)
UserWarning_178760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'UserWarning')

class RankWarning(UserWarning_178760, ):
    str_178761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'str', 'Issued by chebfit when the design matrix is rank deficient.')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 58, 0, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RankWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RankWarning' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'RankWarning', RankWarning)
# Declaration of the 'PolyError' class
# Getting the type of 'Exception' (line 62)
Exception_178762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'Exception')

class PolyError(Exception_178762, ):
    str_178763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'str', 'Base class for errors in this module.')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 62, 0, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PolyError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PolyError' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'PolyError', PolyError)
# Declaration of the 'PolyDomainError' class
# Getting the type of 'PolyError' (line 66)
PolyError_178764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'PolyError')

class PolyDomainError(PolyError_178764, ):
    str_178765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', "Issued by the generic Poly class when two domains don't match.\n\n    This is raised when an binary operation is passed Poly objects with\n    different domains.\n\n    ")
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 66, 0, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PolyDomainError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PolyDomainError' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'PolyDomainError', PolyDomainError)
# Declaration of the 'PolyBase' class

class PolyBase(object, ):
    str_178766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'str', '\n    Base class for all polynomial types.\n\n    Deprecated in numpy 1.9.0, use the abstract\n    ABCPolyBase class instead. Note that the latter\n    reguires a number of virtual functions to be\n    implemented.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 79, 0, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PolyBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PolyBase' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'PolyBase', PolyBase)

@norecursion
def trimseq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'trimseq'
    module_type_store = module_type_store.open_function_context('trimseq', 94, 0, False)
    
    # Passed parameters checking function
    trimseq.stypy_localization = localization
    trimseq.stypy_type_of_self = None
    trimseq.stypy_type_store = module_type_store
    trimseq.stypy_function_name = 'trimseq'
    trimseq.stypy_param_names_list = ['seq']
    trimseq.stypy_varargs_param_name = None
    trimseq.stypy_kwargs_param_name = None
    trimseq.stypy_call_defaults = defaults
    trimseq.stypy_call_varargs = varargs
    trimseq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trimseq', ['seq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trimseq', localization, ['seq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trimseq(...)' code ##################

    str_178767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, (-1)), 'str', 'Remove small Poly series coefficients.\n\n    Parameters\n    ----------\n    seq : sequence\n        Sequence of Poly series coefficients. This routine fails for\n        empty sequences.\n\n    Returns\n    -------\n    series : sequence\n        Subsequence with trailing zeros removed. If the resulting sequence\n        would be empty, return the first element. The returned sequence may\n        or may not be a view.\n\n    Notes\n    -----\n    Do not lose the type info if the sequence contains unknown objects.\n\n    ')
    
    
    
    # Call to len(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'seq' (line 115)
    seq_178769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'seq', False)
    # Processing the call keyword arguments (line 115)
    kwargs_178770 = {}
    # Getting the type of 'len' (line 115)
    len_178768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 7), 'len', False)
    # Calling len(args, kwargs) (line 115)
    len_call_result_178771 = invoke(stypy.reporting.localization.Localization(__file__, 115, 7), len_178768, *[seq_178769], **kwargs_178770)
    
    int_178772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 19), 'int')
    # Applying the binary operator '==' (line 115)
    result_eq_178773 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 7), '==', len_call_result_178771, int_178772)
    
    # Testing the type of an if condition (line 115)
    if_condition_178774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 4), result_eq_178773)
    # Assigning a type to the variable 'if_condition_178774' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'if_condition_178774', if_condition_178774)
    # SSA begins for if statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'seq' (line 116)
    seq_178775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'seq')
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', seq_178775)
    # SSA branch for the else part of an if statement (line 115)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to len(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'seq' (line 118)
    seq_178778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'seq', False)
    # Processing the call keyword arguments (line 118)
    kwargs_178779 = {}
    # Getting the type of 'len' (line 118)
    len_178777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'len', False)
    # Calling len(args, kwargs) (line 118)
    len_call_result_178780 = invoke(stypy.reporting.localization.Localization(__file__, 118, 23), len_178777, *[seq_178778], **kwargs_178779)
    
    int_178781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 34), 'int')
    # Applying the binary operator '-' (line 118)
    result_sub_178782 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 23), '-', len_call_result_178780, int_178781)
    
    int_178783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'int')
    int_178784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 41), 'int')
    # Processing the call keyword arguments (line 118)
    kwargs_178785 = {}
    # Getting the type of 'range' (line 118)
    range_178776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'range', False)
    # Calling range(args, kwargs) (line 118)
    range_call_result_178786 = invoke(stypy.reporting.localization.Localization(__file__, 118, 17), range_178776, *[result_sub_178782, int_178783, int_178784], **kwargs_178785)
    
    # Testing the type of a for loop iterable (line 118)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 8), range_call_result_178786)
    # Getting the type of the for loop variable (line 118)
    for_loop_var_178787 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 8), range_call_result_178786)
    # Assigning a type to the variable 'i' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'i', for_loop_var_178787)
    # SSA begins for a for statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 119)
    i_178788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'i')
    # Getting the type of 'seq' (line 119)
    seq_178789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'seq')
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___178790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), seq_178789, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_178791 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), getitem___178790, i_178788)
    
    int_178792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 25), 'int')
    # Applying the binary operator '!=' (line 119)
    result_ne_178793 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 15), '!=', subscript_call_result_178791, int_178792)
    
    # Testing the type of an if condition (line 119)
    if_condition_178794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 12), result_ne_178793)
    # Assigning a type to the variable 'if_condition_178794' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'if_condition_178794', if_condition_178794)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 121)
    i_178795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'i')
    int_178796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'int')
    # Applying the binary operator '+' (line 121)
    result_add_178797 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 20), '+', i_178795, int_178796)
    
    slice_178798 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 121, 15), None, result_add_178797, None)
    # Getting the type of 'seq' (line 121)
    seq_178799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'seq')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___178800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), seq_178799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_178801 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), getitem___178800, slice_178798)
    
    # Assigning a type to the variable 'stypy_return_type' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', subscript_call_result_178801)
    # SSA join for if statement (line 115)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'trimseq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trimseq' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_178802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_178802)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trimseq'
    return stypy_return_type_178802

# Assigning a type to the variable 'trimseq' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'trimseq', trimseq)

@norecursion
def as_series(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 124)
    True_178803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), 'True')
    defaults = [True_178803]
    # Create a new context for function 'as_series'
    module_type_store = module_type_store.open_function_context('as_series', 124, 0, False)
    
    # Passed parameters checking function
    as_series.stypy_localization = localization
    as_series.stypy_type_of_self = None
    as_series.stypy_type_store = module_type_store
    as_series.stypy_function_name = 'as_series'
    as_series.stypy_param_names_list = ['alist', 'trim']
    as_series.stypy_varargs_param_name = None
    as_series.stypy_kwargs_param_name = None
    as_series.stypy_call_defaults = defaults
    as_series.stypy_call_varargs = varargs
    as_series.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'as_series', ['alist', 'trim'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'as_series', localization, ['alist', 'trim'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'as_series(...)' code ##################

    str_178804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, (-1)), 'str', '\n    Return argument as a list of 1-d arrays.\n\n    The returned list contains array(s) of dtype double, complex double, or\n    object.  A 1-d argument of shape ``(N,)`` is parsed into ``N`` arrays of\n    size one; a 2-d argument of shape ``(M,N)`` is parsed into ``M`` arrays\n    of size ``N`` (i.e., is "parsed by row"); and a higher dimensional array\n    raises a Value Error if it is not first reshaped into either a 1-d or 2-d\n    array.\n\n    Parameters\n    ----------\n    alist : array_like\n        A 1- or 2-d array_like\n    trim : boolean, optional\n        When True, trailing zeros are removed from the inputs.\n        When False, the inputs are passed through intact.\n\n    Returns\n    -------\n    [a1, a2,...] : list of 1-D arrays\n        A copy of the input data as a list of 1-d arrays.\n\n    Raises\n    ------\n    ValueError\n        Raised when `as_series` cannot convert its input to 1-d arrays, or at\n        least one of the resulting arrays is empty.\n\n    Examples\n    --------\n    >>> from numpy import polynomial as P\n    >>> a = np.arange(4)\n    >>> P.as_series(a)\n    [array([ 0.]), array([ 1.]), array([ 2.]), array([ 3.])]\n    >>> b = np.arange(6).reshape((2,3))\n    >>> P.as_series(b)\n    [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.])]\n\n    ')
    
    # Assigning a ListComp to a Name (line 165):
    
    # Assigning a ListComp to a Name (line 165):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'alist' (line 165)
    alist_178814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 52), 'alist')
    comprehension_178815 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), alist_178814)
    # Assigning a type to the variable 'a' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 14), 'a', comprehension_178815)
    
    # Call to array(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'a' (line 165)
    a_178807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'a', False)
    # Processing the call keyword arguments (line 165)
    int_178808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'int')
    keyword_178809 = int_178808
    int_178810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 40), 'int')
    keyword_178811 = int_178810
    kwargs_178812 = {'copy': keyword_178811, 'ndmin': keyword_178809}
    # Getting the type of 'np' (line 165)
    np_178805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 165)
    array_178806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 14), np_178805, 'array')
    # Calling array(args, kwargs) (line 165)
    array_call_result_178813 = invoke(stypy.reporting.localization.Localization(__file__, 165, 14), array_178806, *[a_178807], **kwargs_178812)
    
    list_178816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_178816, array_call_result_178813)
    # Assigning a type to the variable 'arrays' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'arrays', list_178816)
    
    
    
    # Call to min(...): (line 166)
    # Processing the call arguments (line 166)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 166)
    arrays_178820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'arrays', False)
    comprehension_178821 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 12), arrays_178820)
    # Assigning a type to the variable 'a' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'a', comprehension_178821)
    # Getting the type of 'a' (line 166)
    a_178818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'a', False)
    # Obtaining the member 'size' of a type (line 166)
    size_178819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), a_178818, 'size')
    list_178822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 12), list_178822, size_178819)
    # Processing the call keyword arguments (line 166)
    kwargs_178823 = {}
    # Getting the type of 'min' (line 166)
    min_178817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 7), 'min', False)
    # Calling min(args, kwargs) (line 166)
    min_call_result_178824 = invoke(stypy.reporting.localization.Localization(__file__, 166, 7), min_178817, *[list_178822], **kwargs_178823)
    
    int_178825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 40), 'int')
    # Applying the binary operator '==' (line 166)
    result_eq_178826 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 7), '==', min_call_result_178824, int_178825)
    
    # Testing the type of an if condition (line 166)
    if_condition_178827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 4), result_eq_178826)
    # Assigning a type to the variable 'if_condition_178827' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'if_condition_178827', if_condition_178827)
    # SSA begins for if statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 167)
    # Processing the call arguments (line 167)
    str_178829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'str', 'Coefficient array is empty')
    # Processing the call keyword arguments (line 167)
    kwargs_178830 = {}
    # Getting the type of 'ValueError' (line 167)
    ValueError_178828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 167)
    ValueError_call_result_178831 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), ValueError_178828, *[str_178829], **kwargs_178830)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 167, 8), ValueError_call_result_178831, 'raise parameter', BaseException)
    # SSA join for if statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 168)
    # Processing the call arguments (line 168)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 168)
    arrays_178837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 33), 'arrays', False)
    comprehension_178838 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), arrays_178837)
    # Assigning a type to the variable 'a' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'a', comprehension_178838)
    
    # Getting the type of 'a' (line 168)
    a_178833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'a', False)
    # Obtaining the member 'ndim' of a type (line 168)
    ndim_178834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), a_178833, 'ndim')
    int_178835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 22), 'int')
    # Applying the binary operator '!=' (line 168)
    result_ne_178836 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 12), '!=', ndim_178834, int_178835)
    
    list_178839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_178839, result_ne_178836)
    # Processing the call keyword arguments (line 168)
    kwargs_178840 = {}
    # Getting the type of 'any' (line 168)
    any_178832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 7), 'any', False)
    # Calling any(args, kwargs) (line 168)
    any_call_result_178841 = invoke(stypy.reporting.localization.Localization(__file__, 168, 7), any_178832, *[list_178839], **kwargs_178840)
    
    # Testing the type of an if condition (line 168)
    if_condition_178842 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 4), any_call_result_178841)
    # Assigning a type to the variable 'if_condition_178842' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'if_condition_178842', if_condition_178842)
    # SSA begins for if statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 169)
    # Processing the call arguments (line 169)
    str_178844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 25), 'str', 'Coefficient array is not 1-d')
    # Processing the call keyword arguments (line 169)
    kwargs_178845 = {}
    # Getting the type of 'ValueError' (line 169)
    ValueError_178843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 169)
    ValueError_call_result_178846 = invoke(stypy.reporting.localization.Localization(__file__, 169, 14), ValueError_178843, *[str_178844], **kwargs_178845)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 169, 8), ValueError_call_result_178846, 'raise parameter', BaseException)
    # SSA join for if statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'trim' (line 170)
    trim_178847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 7), 'trim')
    # Testing the type of an if condition (line 170)
    if_condition_178848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), trim_178847)
    # Assigning a type to the variable 'if_condition_178848' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_178848', if_condition_178848)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Name (line 171):
    
    # Assigning a ListComp to a Name (line 171):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 171)
    arrays_178853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 38), 'arrays')
    comprehension_178854 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 18), arrays_178853)
    # Assigning a type to the variable 'a' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'a', comprehension_178854)
    
    # Call to trimseq(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'a' (line 171)
    a_178850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'a', False)
    # Processing the call keyword arguments (line 171)
    kwargs_178851 = {}
    # Getting the type of 'trimseq' (line 171)
    trimseq_178849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'trimseq', False)
    # Calling trimseq(args, kwargs) (line 171)
    trimseq_call_result_178852 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), trimseq_178849, *[a_178850], **kwargs_178851)
    
    list_178855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 18), list_178855, trimseq_call_result_178852)
    # Assigning a type to the variable 'arrays' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'arrays', list_178855)
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 173)
    # Processing the call arguments (line 173)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 173)
    arrays_178865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 49), 'arrays', False)
    comprehension_178866 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 12), arrays_178865)
    # Assigning a type to the variable 'a' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'a', comprehension_178866)
    
    # Getting the type of 'a' (line 173)
    a_178857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'a', False)
    # Obtaining the member 'dtype' of a type (line 173)
    dtype_178858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 12), a_178857, 'dtype')
    
    # Call to dtype(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'object' (line 173)
    object_178861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'object', False)
    # Processing the call keyword arguments (line 173)
    kwargs_178862 = {}
    # Getting the type of 'np' (line 173)
    np_178859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'np', False)
    # Obtaining the member 'dtype' of a type (line 173)
    dtype_178860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 23), np_178859, 'dtype')
    # Calling dtype(args, kwargs) (line 173)
    dtype_call_result_178863 = invoke(stypy.reporting.localization.Localization(__file__, 173, 23), dtype_178860, *[object_178861], **kwargs_178862)
    
    # Applying the binary operator '==' (line 173)
    result_eq_178864 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 12), '==', dtype_178858, dtype_call_result_178863)
    
    list_178867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 12), list_178867, result_eq_178864)
    # Processing the call keyword arguments (line 173)
    kwargs_178868 = {}
    # Getting the type of 'any' (line 173)
    any_178856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 7), 'any', False)
    # Calling any(args, kwargs) (line 173)
    any_call_result_178869 = invoke(stypy.reporting.localization.Localization(__file__, 173, 7), any_178856, *[list_178867], **kwargs_178868)
    
    # Testing the type of an if condition (line 173)
    if_condition_178870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 4), any_call_result_178869)
    # Assigning a type to the variable 'if_condition_178870' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'if_condition_178870', if_condition_178870)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 174):
    
    # Assigning a List to a Name (line 174):
    
    # Obtaining an instance of the builtin type 'list' (line 174)
    list_178871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 174)
    
    # Assigning a type to the variable 'ret' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'ret', list_178871)
    
    # Getting the type of 'arrays' (line 175)
    arrays_178872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'arrays')
    # Testing the type of a for loop iterable (line 175)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 175, 8), arrays_178872)
    # Getting the type of the for loop variable (line 175)
    for_loop_var_178873 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 175, 8), arrays_178872)
    # Assigning a type to the variable 'a' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'a', for_loop_var_178873)
    # SSA begins for a for statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 176)
    a_178874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'a')
    # Obtaining the member 'dtype' of a type (line 176)
    dtype_178875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), a_178874, 'dtype')
    
    # Call to dtype(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'object' (line 176)
    object_178878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'object', False)
    # Processing the call keyword arguments (line 176)
    kwargs_178879 = {}
    # Getting the type of 'np' (line 176)
    np_178876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'np', False)
    # Obtaining the member 'dtype' of a type (line 176)
    dtype_178877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 26), np_178876, 'dtype')
    # Calling dtype(args, kwargs) (line 176)
    dtype_call_result_178880 = invoke(stypy.reporting.localization.Localization(__file__, 176, 26), dtype_178877, *[object_178878], **kwargs_178879)
    
    # Applying the binary operator '!=' (line 176)
    result_ne_178881 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 15), '!=', dtype_178875, dtype_call_result_178880)
    
    # Testing the type of an if condition (line 176)
    if_condition_178882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 12), result_ne_178881)
    # Assigning a type to the variable 'if_condition_178882' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'if_condition_178882', if_condition_178882)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to empty(...): (line 177)
    # Processing the call arguments (line 177)
    
    # Call to len(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'a' (line 177)
    a_178886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'a', False)
    # Processing the call keyword arguments (line 177)
    kwargs_178887 = {}
    # Getting the type of 'len' (line 177)
    len_178885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'len', False)
    # Calling len(args, kwargs) (line 177)
    len_call_result_178888 = invoke(stypy.reporting.localization.Localization(__file__, 177, 31), len_178885, *[a_178886], **kwargs_178887)
    
    # Processing the call keyword arguments (line 177)
    
    # Call to dtype(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'object' (line 177)
    object_178891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 54), 'object', False)
    # Processing the call keyword arguments (line 177)
    kwargs_178892 = {}
    # Getting the type of 'np' (line 177)
    np_178889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 45), 'np', False)
    # Obtaining the member 'dtype' of a type (line 177)
    dtype_178890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 45), np_178889, 'dtype')
    # Calling dtype(args, kwargs) (line 177)
    dtype_call_result_178893 = invoke(stypy.reporting.localization.Localization(__file__, 177, 45), dtype_178890, *[object_178891], **kwargs_178892)
    
    keyword_178894 = dtype_call_result_178893
    kwargs_178895 = {'dtype': keyword_178894}
    # Getting the type of 'np' (line 177)
    np_178883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 22), 'np', False)
    # Obtaining the member 'empty' of a type (line 177)
    empty_178884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 22), np_178883, 'empty')
    # Calling empty(args, kwargs) (line 177)
    empty_call_result_178896 = invoke(stypy.reporting.localization.Localization(__file__, 177, 22), empty_178884, *[len_call_result_178888], **kwargs_178895)
    
    # Assigning a type to the variable 'tmp' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'tmp', empty_call_result_178896)
    
    # Assigning a Subscript to a Subscript (line 178):
    
    # Assigning a Subscript to a Subscript (line 178):
    
    # Obtaining the type of the subscript
    slice_178897 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 25), None, None, None)
    # Getting the type of 'a' (line 178)
    a_178898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'a')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___178899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 25), a_178898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_178900 = invoke(stypy.reporting.localization.Localization(__file__, 178, 25), getitem___178899, slice_178897)
    
    # Getting the type of 'tmp' (line 178)
    tmp_178901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'tmp')
    slice_178902 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 16), None, None, None)
    # Storing an element on a container (line 178)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), tmp_178901, (slice_178902, subscript_call_result_178900))
    
    # Call to append(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'tmp' (line 179)
    tmp_178905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'tmp', False)
    # Processing the call keyword arguments (line 179)
    kwargs_178906 = {}
    # Getting the type of 'ret' (line 179)
    ret_178903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'ret', False)
    # Obtaining the member 'append' of a type (line 179)
    append_178904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), ret_178903, 'append')
    # Calling append(args, kwargs) (line 179)
    append_call_result_178907 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), append_178904, *[tmp_178905], **kwargs_178906)
    
    # SSA branch for the else part of an if statement (line 176)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 181)
    # Processing the call arguments (line 181)
    
    # Call to copy(...): (line 181)
    # Processing the call keyword arguments (line 181)
    kwargs_178912 = {}
    # Getting the type of 'a' (line 181)
    a_178910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 27), 'a', False)
    # Obtaining the member 'copy' of a type (line 181)
    copy_178911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 27), a_178910, 'copy')
    # Calling copy(args, kwargs) (line 181)
    copy_call_result_178913 = invoke(stypy.reporting.localization.Localization(__file__, 181, 27), copy_178911, *[], **kwargs_178912)
    
    # Processing the call keyword arguments (line 181)
    kwargs_178914 = {}
    # Getting the type of 'ret' (line 181)
    ret_178908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'ret', False)
    # Obtaining the member 'append' of a type (line 181)
    append_178909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), ret_178908, 'append')
    # Calling append(args, kwargs) (line 181)
    append_call_result_178915 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), append_178909, *[copy_call_result_178913], **kwargs_178914)
    
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 173)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to common_type(...): (line 184)
    # Getting the type of 'arrays' (line 184)
    arrays_178918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'arrays', False)
    # Processing the call keyword arguments (line 184)
    kwargs_178919 = {}
    # Getting the type of 'np' (line 184)
    np_178916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'np', False)
    # Obtaining the member 'common_type' of a type (line 184)
    common_type_178917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 20), np_178916, 'common_type')
    # Calling common_type(args, kwargs) (line 184)
    common_type_call_result_178920 = invoke(stypy.reporting.localization.Localization(__file__, 184, 20), common_type_178917, *[arrays_178918], **kwargs_178919)
    
    # Assigning a type to the variable 'dtype' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'dtype', common_type_call_result_178920)
    # SSA branch for the except part of a try statement (line 183)
    # SSA branch for the except '<any exception>' branch of a try statement (line 183)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 186)
    # Processing the call arguments (line 186)
    str_178922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 29), 'str', 'Coefficient arrays have no common type')
    # Processing the call keyword arguments (line 186)
    kwargs_178923 = {}
    # Getting the type of 'ValueError' (line 186)
    ValueError_178921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 186)
    ValueError_call_result_178924 = invoke(stypy.reporting.localization.Localization(__file__, 186, 18), ValueError_178921, *[str_178922], **kwargs_178923)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 186, 12), ValueError_call_result_178924, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 187):
    
    # Assigning a ListComp to a Name (line 187):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 187)
    arrays_178934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 57), 'arrays')
    comprehension_178935 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), arrays_178934)
    # Assigning a type to the variable 'a' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'a', comprehension_178935)
    
    # Call to array(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'a' (line 187)
    a_178927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'a', False)
    # Processing the call keyword arguments (line 187)
    int_178928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 32), 'int')
    keyword_178929 = int_178928
    # Getting the type of 'dtype' (line 187)
    dtype_178930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 41), 'dtype', False)
    keyword_178931 = dtype_178930
    kwargs_178932 = {'dtype': keyword_178931, 'copy': keyword_178929}
    # Getting the type of 'np' (line 187)
    np_178925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 187)
    array_178926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), np_178925, 'array')
    # Calling array(args, kwargs) (line 187)
    array_call_result_178933 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), array_178926, *[a_178927], **kwargs_178932)
    
    list_178936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), list_178936, array_call_result_178933)
    # Assigning a type to the variable 'ret' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'ret', list_178936)
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 188)
    ret_178937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type', ret_178937)
    
    # ################# End of 'as_series(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'as_series' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_178938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_178938)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'as_series'
    return stypy_return_type_178938

# Assigning a type to the variable 'as_series' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'as_series', as_series)

@norecursion
def trimcoef(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_178939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 20), 'int')
    defaults = [int_178939]
    # Create a new context for function 'trimcoef'
    module_type_store = module_type_store.open_function_context('trimcoef', 191, 0, False)
    
    # Passed parameters checking function
    trimcoef.stypy_localization = localization
    trimcoef.stypy_type_of_self = None
    trimcoef.stypy_type_store = module_type_store
    trimcoef.stypy_function_name = 'trimcoef'
    trimcoef.stypy_param_names_list = ['c', 'tol']
    trimcoef.stypy_varargs_param_name = None
    trimcoef.stypy_kwargs_param_name = None
    trimcoef.stypy_call_defaults = defaults
    trimcoef.stypy_call_varargs = varargs
    trimcoef.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trimcoef', ['c', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trimcoef', localization, ['c', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trimcoef(...)' code ##################

    str_178940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, (-1)), 'str', '\n    Remove "small" "trailing" coefficients from a polynomial.\n\n    "Small" means "small in absolute value" and is controlled by the\n    parameter `tol`; "trailing" means highest order coefficient(s), e.g., in\n    ``[0, 1, 1, 0, 0]`` (which represents ``0 + x + x**2 + 0*x**3 + 0*x**4``)\n    both the 3-rd and 4-th order coefficients would be "trimmed."\n\n    Parameters\n    ----------\n    c : array_like\n        1-d array of coefficients, ordered from lowest order to highest.\n    tol : number, optional\n        Trailing (i.e., highest order) elements with absolute value less\n        than or equal to `tol` (default value is zero) are removed.\n\n    Returns\n    -------\n    trimmed : ndarray\n        1-d array with trailing zeros removed.  If the resulting series\n        would be empty, a series containing a single zero is returned.\n\n    Raises\n    ------\n    ValueError\n        If `tol` < 0\n\n    See Also\n    --------\n    trimseq\n\n    Examples\n    --------\n    >>> from numpy import polynomial as P\n    >>> P.trimcoef((0,0,3,0,5,0,0))\n    array([ 0.,  0.,  3.,  0.,  5.])\n    >>> P.trimcoef((0,0,1e-3,0,1e-5,0,0),1e-3) # item == tol is trimmed\n    array([ 0.])\n    >>> i = complex(0,1) # works for complex\n    >>> P.trimcoef((3e-4,1e-3*(1-i),5e-4,2e-5*(1+i)), 1e-3)\n    array([ 0.0003+0.j   ,  0.0010-0.001j])\n\n    ')
    
    
    # Getting the type of 'tol' (line 235)
    tol_178941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'tol')
    int_178942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 13), 'int')
    # Applying the binary operator '<' (line 235)
    result_lt_178943 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 7), '<', tol_178941, int_178942)
    
    # Testing the type of an if condition (line 235)
    if_condition_178944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 4), result_lt_178943)
    # Assigning a type to the variable 'if_condition_178944' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'if_condition_178944', if_condition_178944)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 236)
    # Processing the call arguments (line 236)
    str_178946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 25), 'str', 'tol must be non-negative')
    # Processing the call keyword arguments (line 236)
    kwargs_178947 = {}
    # Getting the type of 'ValueError' (line 236)
    ValueError_178945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 236)
    ValueError_call_result_178948 = invoke(stypy.reporting.localization.Localization(__file__, 236, 14), ValueError_178945, *[str_178946], **kwargs_178947)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 236, 8), ValueError_call_result_178948, 'raise parameter', BaseException)
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a List (line 238):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 238)
    # Processing the call arguments (line 238)
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_178950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    # Getting the type of 'c' (line 238)
    c_178951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 20), list_178950, c_178951)
    
    # Processing the call keyword arguments (line 238)
    kwargs_178952 = {}
    # Getting the type of 'as_series' (line 238)
    as_series_178949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 10), 'as_series', False)
    # Calling as_series(args, kwargs) (line 238)
    as_series_call_result_178953 = invoke(stypy.reporting.localization.Localization(__file__, 238, 10), as_series_178949, *[list_178950], **kwargs_178952)
    
    # Assigning a type to the variable 'call_assignment_178733' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'call_assignment_178733', as_series_call_result_178953)
    
    # Assigning a Call to a Name (line 238):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_178956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 4), 'int')
    # Processing the call keyword arguments
    kwargs_178957 = {}
    # Getting the type of 'call_assignment_178733' (line 238)
    call_assignment_178733_178954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'call_assignment_178733', False)
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___178955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 4), call_assignment_178733_178954, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_178958 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178955, *[int_178956], **kwargs_178957)
    
    # Assigning a type to the variable 'call_assignment_178734' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'call_assignment_178734', getitem___call_result_178958)
    
    # Assigning a Name to a Name (line 238):
    # Getting the type of 'call_assignment_178734' (line 238)
    call_assignment_178734_178959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'call_assignment_178734')
    # Assigning a type to the variable 'c' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 5), 'c', call_assignment_178734_178959)
    
    # Assigning a Call to a List (line 239):
    
    # Assigning a Call to a Name:
    
    # Call to where(...): (line 239)
    # Processing the call arguments (line 239)
    
    
    # Call to abs(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'c' (line 239)
    c_178964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'c', False)
    # Processing the call keyword arguments (line 239)
    kwargs_178965 = {}
    # Getting the type of 'np' (line 239)
    np_178962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'np', False)
    # Obtaining the member 'abs' of a type (line 239)
    abs_178963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 21), np_178962, 'abs')
    # Calling abs(args, kwargs) (line 239)
    abs_call_result_178966 = invoke(stypy.reporting.localization.Localization(__file__, 239, 21), abs_178963, *[c_178964], **kwargs_178965)
    
    # Getting the type of 'tol' (line 239)
    tol_178967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 33), 'tol', False)
    # Applying the binary operator '>' (line 239)
    result_gt_178968 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 21), '>', abs_call_result_178966, tol_178967)
    
    # Processing the call keyword arguments (line 239)
    kwargs_178969 = {}
    # Getting the type of 'np' (line 239)
    np_178960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'np', False)
    # Obtaining the member 'where' of a type (line 239)
    where_178961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), np_178960, 'where')
    # Calling where(args, kwargs) (line 239)
    where_call_result_178970 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), where_178961, *[result_gt_178968], **kwargs_178969)
    
    # Assigning a type to the variable 'call_assignment_178735' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_178735', where_call_result_178970)
    
    # Assigning a Call to a Name (line 239):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_178973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 4), 'int')
    # Processing the call keyword arguments
    kwargs_178974 = {}
    # Getting the type of 'call_assignment_178735' (line 239)
    call_assignment_178735_178971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_178735', False)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___178972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 4), call_assignment_178735_178971, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_178975 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178972, *[int_178973], **kwargs_178974)
    
    # Assigning a type to the variable 'call_assignment_178736' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_178736', getitem___call_result_178975)
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'call_assignment_178736' (line 239)
    call_assignment_178736_178976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_178736')
    # Assigning a type to the variable 'ind' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 5), 'ind', call_assignment_178736_178976)
    
    
    
    # Call to len(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'ind' (line 240)
    ind_178978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'ind', False)
    # Processing the call keyword arguments (line 240)
    kwargs_178979 = {}
    # Getting the type of 'len' (line 240)
    len_178977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 'len', False)
    # Calling len(args, kwargs) (line 240)
    len_call_result_178980 = invoke(stypy.reporting.localization.Localization(__file__, 240, 7), len_178977, *[ind_178978], **kwargs_178979)
    
    int_178981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 19), 'int')
    # Applying the binary operator '==' (line 240)
    result_eq_178982 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 7), '==', len_call_result_178980, int_178981)
    
    # Testing the type of an if condition (line 240)
    if_condition_178983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 4), result_eq_178982)
    # Assigning a type to the variable 'if_condition_178983' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'if_condition_178983', if_condition_178983)
    # SSA begins for if statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_178984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 18), 'int')
    slice_178985 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 15), None, int_178984, None)
    # Getting the type of 'c' (line 241)
    c_178986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'c')
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___178987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 15), c_178986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_178988 = invoke(stypy.reporting.localization.Localization(__file__, 241, 15), getitem___178987, slice_178985)
    
    int_178989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 21), 'int')
    # Applying the binary operator '*' (line 241)
    result_mul_178990 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 15), '*', subscript_call_result_178988, int_178989)
    
    # Assigning a type to the variable 'stypy_return_type' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'stypy_return_type', result_mul_178990)
    # SSA branch for the else part of an if statement (line 240)
    module_type_store.open_ssa_branch('else')
    
    # Call to copy(...): (line 243)
    # Processing the call keyword arguments (line 243)
    kwargs_179002 = {}
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_178991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 22), 'int')
    # Getting the type of 'ind' (line 243)
    ind_178992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'ind', False)
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___178993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 18), ind_178992, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_178994 = invoke(stypy.reporting.localization.Localization(__file__, 243, 18), getitem___178993, int_178991)
    
    int_178995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 28), 'int')
    # Applying the binary operator '+' (line 243)
    result_add_178996 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 18), '+', subscript_call_result_178994, int_178995)
    
    slice_178997 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 243, 15), None, result_add_178996, None)
    # Getting the type of 'c' (line 243)
    c_178998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___178999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 15), c_178998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_179000 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), getitem___178999, slice_178997)
    
    # Obtaining the member 'copy' of a type (line 243)
    copy_179001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 15), subscript_call_result_179000, 'copy')
    # Calling copy(args, kwargs) (line 243)
    copy_call_result_179003 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), copy_179001, *[], **kwargs_179002)
    
    # Assigning a type to the variable 'stypy_return_type' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'stypy_return_type', copy_call_result_179003)
    # SSA join for if statement (line 240)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'trimcoef(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trimcoef' in the type store
    # Getting the type of 'stypy_return_type' (line 191)
    stypy_return_type_179004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_179004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trimcoef'
    return stypy_return_type_179004

# Assigning a type to the variable 'trimcoef' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'trimcoef', trimcoef)

@norecursion
def getdomain(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getdomain'
    module_type_store = module_type_store.open_function_context('getdomain', 245, 0, False)
    
    # Passed parameters checking function
    getdomain.stypy_localization = localization
    getdomain.stypy_type_of_self = None
    getdomain.stypy_type_store = module_type_store
    getdomain.stypy_function_name = 'getdomain'
    getdomain.stypy_param_names_list = ['x']
    getdomain.stypy_varargs_param_name = None
    getdomain.stypy_kwargs_param_name = None
    getdomain.stypy_call_defaults = defaults
    getdomain.stypy_call_varargs = varargs
    getdomain.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getdomain', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getdomain', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getdomain(...)' code ##################

    str_179005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, (-1)), 'str', '\n    Return a domain suitable for given abscissae.\n\n    Find a domain suitable for a polynomial or Chebyshev series\n    defined at the values supplied.\n\n    Parameters\n    ----------\n    x : array_like\n        1-d array of abscissae whose domain will be determined.\n\n    Returns\n    -------\n    domain : ndarray\n        1-d array containing two values.  If the inputs are complex, then\n        the two returned points are the lower left and upper right corners\n        of the smallest rectangle (aligned with the axes) in the complex\n        plane containing the points `x`. If the inputs are real, then the\n        two points are the ends of the smallest interval containing the\n        points `x`.\n\n    See Also\n    --------\n    mapparms, mapdomain\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polyutils as pu\n    >>> points = np.arange(4)**2 - 5; points\n    array([-5, -4, -1,  4])\n    >>> pu.getdomain(points)\n    array([-5.,  4.])\n    >>> c = np.exp(complex(0,1)*np.pi*np.arange(12)/6) # unit circle\n    >>> pu.getdomain(c)\n    array([-1.-1.j,  1.+1.j])\n\n    ')
    
    # Assigning a Call to a List (line 283):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 283)
    # Processing the call arguments (line 283)
    
    # Obtaining an instance of the builtin type 'list' (line 283)
    list_179007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 283)
    # Adding element type (line 283)
    # Getting the type of 'x' (line 283)
    x_179008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 20), list_179007, x_179008)
    
    # Processing the call keyword arguments (line 283)
    # Getting the type of 'False' (line 283)
    False_179009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 30), 'False', False)
    keyword_179010 = False_179009
    kwargs_179011 = {'trim': keyword_179010}
    # Getting the type of 'as_series' (line 283)
    as_series_179006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 10), 'as_series', False)
    # Calling as_series(args, kwargs) (line 283)
    as_series_call_result_179012 = invoke(stypy.reporting.localization.Localization(__file__, 283, 10), as_series_179006, *[list_179007], **kwargs_179011)
    
    # Assigning a type to the variable 'call_assignment_178737' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'call_assignment_178737', as_series_call_result_179012)
    
    # Assigning a Call to a Name (line 283):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_179015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 4), 'int')
    # Processing the call keyword arguments
    kwargs_179016 = {}
    # Getting the type of 'call_assignment_178737' (line 283)
    call_assignment_178737_179013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'call_assignment_178737', False)
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___179014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 4), call_assignment_178737_179013, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_179017 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179014, *[int_179015], **kwargs_179016)
    
    # Assigning a type to the variable 'call_assignment_178738' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'call_assignment_178738', getitem___call_result_179017)
    
    # Assigning a Name to a Name (line 283):
    # Getting the type of 'call_assignment_178738' (line 283)
    call_assignment_178738_179018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'call_assignment_178738')
    # Assigning a type to the variable 'x' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 5), 'x', call_assignment_178738_179018)
    
    
    # Getting the type of 'x' (line 284)
    x_179019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 7), 'x')
    # Obtaining the member 'dtype' of a type (line 284)
    dtype_179020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 7), x_179019, 'dtype')
    # Obtaining the member 'char' of a type (line 284)
    char_179021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 7), dtype_179020, 'char')
    
    # Obtaining the type of the subscript
    str_179022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 36), 'str', 'Complex')
    # Getting the type of 'np' (line 284)
    np_179023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'np')
    # Obtaining the member 'typecodes' of a type (line 284)
    typecodes_179024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 23), np_179023, 'typecodes')
    # Obtaining the member '__getitem__' of a type (line 284)
    getitem___179025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 23), typecodes_179024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 284)
    subscript_call_result_179026 = invoke(stypy.reporting.localization.Localization(__file__, 284, 23), getitem___179025, str_179022)
    
    # Applying the binary operator 'in' (line 284)
    result_contains_179027 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 7), 'in', char_179021, subscript_call_result_179026)
    
    # Testing the type of an if condition (line 284)
    if_condition_179028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 4), result_contains_179027)
    # Assigning a type to the variable 'if_condition_179028' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'if_condition_179028', if_condition_179028)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 285):
    
    # Assigning a Call to a Name (line 285):
    
    # Call to min(...): (line 285)
    # Processing the call keyword arguments (line 285)
    kwargs_179032 = {}
    # Getting the type of 'x' (line 285)
    x_179029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'x', False)
    # Obtaining the member 'real' of a type (line 285)
    real_179030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 21), x_179029, 'real')
    # Obtaining the member 'min' of a type (line 285)
    min_179031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 21), real_179030, 'min')
    # Calling min(args, kwargs) (line 285)
    min_call_result_179033 = invoke(stypy.reporting.localization.Localization(__file__, 285, 21), min_179031, *[], **kwargs_179032)
    
    # Assigning a type to the variable 'tuple_assignment_178739' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_assignment_178739', min_call_result_179033)
    
    # Assigning a Call to a Name (line 285):
    
    # Call to max(...): (line 285)
    # Processing the call keyword arguments (line 285)
    kwargs_179037 = {}
    # Getting the type of 'x' (line 285)
    x_179034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 35), 'x', False)
    # Obtaining the member 'real' of a type (line 285)
    real_179035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 35), x_179034, 'real')
    # Obtaining the member 'max' of a type (line 285)
    max_179036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 35), real_179035, 'max')
    # Calling max(args, kwargs) (line 285)
    max_call_result_179038 = invoke(stypy.reporting.localization.Localization(__file__, 285, 35), max_179036, *[], **kwargs_179037)
    
    # Assigning a type to the variable 'tuple_assignment_178740' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_assignment_178740', max_call_result_179038)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'tuple_assignment_178739' (line 285)
    tuple_assignment_178739_179039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_assignment_178739')
    # Assigning a type to the variable 'rmin' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'rmin', tuple_assignment_178739_179039)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'tuple_assignment_178740' (line 285)
    tuple_assignment_178740_179040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_assignment_178740')
    # Assigning a type to the variable 'rmax' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 14), 'rmax', tuple_assignment_178740_179040)
    
    # Assigning a Tuple to a Tuple (line 286):
    
    # Assigning a Call to a Name (line 286):
    
    # Call to min(...): (line 286)
    # Processing the call keyword arguments (line 286)
    kwargs_179044 = {}
    # Getting the type of 'x' (line 286)
    x_179041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 21), 'x', False)
    # Obtaining the member 'imag' of a type (line 286)
    imag_179042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 21), x_179041, 'imag')
    # Obtaining the member 'min' of a type (line 286)
    min_179043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 21), imag_179042, 'min')
    # Calling min(args, kwargs) (line 286)
    min_call_result_179045 = invoke(stypy.reporting.localization.Localization(__file__, 286, 21), min_179043, *[], **kwargs_179044)
    
    # Assigning a type to the variable 'tuple_assignment_178741' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_assignment_178741', min_call_result_179045)
    
    # Assigning a Call to a Name (line 286):
    
    # Call to max(...): (line 286)
    # Processing the call keyword arguments (line 286)
    kwargs_179049 = {}
    # Getting the type of 'x' (line 286)
    x_179046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 35), 'x', False)
    # Obtaining the member 'imag' of a type (line 286)
    imag_179047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 35), x_179046, 'imag')
    # Obtaining the member 'max' of a type (line 286)
    max_179048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 35), imag_179047, 'max')
    # Calling max(args, kwargs) (line 286)
    max_call_result_179050 = invoke(stypy.reporting.localization.Localization(__file__, 286, 35), max_179048, *[], **kwargs_179049)
    
    # Assigning a type to the variable 'tuple_assignment_178742' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_assignment_178742', max_call_result_179050)
    
    # Assigning a Name to a Name (line 286):
    # Getting the type of 'tuple_assignment_178741' (line 286)
    tuple_assignment_178741_179051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_assignment_178741')
    # Assigning a type to the variable 'imin' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'imin', tuple_assignment_178741_179051)
    
    # Assigning a Name to a Name (line 286):
    # Getting the type of 'tuple_assignment_178742' (line 286)
    tuple_assignment_178742_179052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_assignment_178742')
    # Assigning a type to the variable 'imax' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 14), 'imax', tuple_assignment_178742_179052)
    
    # Call to array(...): (line 287)
    # Processing the call arguments (line 287)
    
    # Obtaining an instance of the builtin type 'tuple' (line 287)
    tuple_179055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 287)
    # Adding element type (line 287)
    
    # Call to complex(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'rmin' (line 287)
    rmin_179057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 33), 'rmin', False)
    # Getting the type of 'imin' (line 287)
    imin_179058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 39), 'imin', False)
    # Processing the call keyword arguments (line 287)
    kwargs_179059 = {}
    # Getting the type of 'complex' (line 287)
    complex_179056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'complex', False)
    # Calling complex(args, kwargs) (line 287)
    complex_call_result_179060 = invoke(stypy.reporting.localization.Localization(__file__, 287, 25), complex_179056, *[rmin_179057, imin_179058], **kwargs_179059)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 25), tuple_179055, complex_call_result_179060)
    # Adding element type (line 287)
    
    # Call to complex(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'rmax' (line 287)
    rmax_179062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 54), 'rmax', False)
    # Getting the type of 'imax' (line 287)
    imax_179063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'imax', False)
    # Processing the call keyword arguments (line 287)
    kwargs_179064 = {}
    # Getting the type of 'complex' (line 287)
    complex_179061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 46), 'complex', False)
    # Calling complex(args, kwargs) (line 287)
    complex_call_result_179065 = invoke(stypy.reporting.localization.Localization(__file__, 287, 46), complex_179061, *[rmax_179062, imax_179063], **kwargs_179064)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 25), tuple_179055, complex_call_result_179065)
    
    # Processing the call keyword arguments (line 287)
    kwargs_179066 = {}
    # Getting the type of 'np' (line 287)
    np_179053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 287)
    array_179054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 15), np_179053, 'array')
    # Calling array(args, kwargs) (line 287)
    array_call_result_179067 = invoke(stypy.reporting.localization.Localization(__file__, 287, 15), array_179054, *[tuple_179055], **kwargs_179066)
    
    # Assigning a type to the variable 'stypy_return_type' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'stypy_return_type', array_call_result_179067)
    # SSA branch for the else part of an if statement (line 284)
    module_type_store.open_ssa_branch('else')
    
    # Call to array(...): (line 289)
    # Processing the call arguments (line 289)
    
    # Obtaining an instance of the builtin type 'tuple' (line 289)
    tuple_179070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 289)
    # Adding element type (line 289)
    
    # Call to min(...): (line 289)
    # Processing the call keyword arguments (line 289)
    kwargs_179073 = {}
    # Getting the type of 'x' (line 289)
    x_179071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 25), 'x', False)
    # Obtaining the member 'min' of a type (line 289)
    min_179072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 25), x_179071, 'min')
    # Calling min(args, kwargs) (line 289)
    min_call_result_179074 = invoke(stypy.reporting.localization.Localization(__file__, 289, 25), min_179072, *[], **kwargs_179073)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 25), tuple_179070, min_call_result_179074)
    # Adding element type (line 289)
    
    # Call to max(...): (line 289)
    # Processing the call keyword arguments (line 289)
    kwargs_179077 = {}
    # Getting the type of 'x' (line 289)
    x_179075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 34), 'x', False)
    # Obtaining the member 'max' of a type (line 289)
    max_179076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 34), x_179075, 'max')
    # Calling max(args, kwargs) (line 289)
    max_call_result_179078 = invoke(stypy.reporting.localization.Localization(__file__, 289, 34), max_179076, *[], **kwargs_179077)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 25), tuple_179070, max_call_result_179078)
    
    # Processing the call keyword arguments (line 289)
    kwargs_179079 = {}
    # Getting the type of 'np' (line 289)
    np_179068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 289)
    array_179069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 15), np_179068, 'array')
    # Calling array(args, kwargs) (line 289)
    array_call_result_179080 = invoke(stypy.reporting.localization.Localization(__file__, 289, 15), array_179069, *[tuple_179070], **kwargs_179079)
    
    # Assigning a type to the variable 'stypy_return_type' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type', array_call_result_179080)
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'getdomain(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getdomain' in the type store
    # Getting the type of 'stypy_return_type' (line 245)
    stypy_return_type_179081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_179081)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getdomain'
    return stypy_return_type_179081

# Assigning a type to the variable 'getdomain' (line 245)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'getdomain', getdomain)

@norecursion
def mapparms(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mapparms'
    module_type_store = module_type_store.open_function_context('mapparms', 291, 0, False)
    
    # Passed parameters checking function
    mapparms.stypy_localization = localization
    mapparms.stypy_type_of_self = None
    mapparms.stypy_type_store = module_type_store
    mapparms.stypy_function_name = 'mapparms'
    mapparms.stypy_param_names_list = ['old', 'new']
    mapparms.stypy_varargs_param_name = None
    mapparms.stypy_kwargs_param_name = None
    mapparms.stypy_call_defaults = defaults
    mapparms.stypy_call_varargs = varargs
    mapparms.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mapparms', ['old', 'new'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mapparms', localization, ['old', 'new'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mapparms(...)' code ##################

    str_179082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, (-1)), 'str', '\n    Linear map parameters between domains.\n\n    Return the parameters of the linear map ``offset + scale*x`` that maps\n    `old` to `new` such that ``old[i] -> new[i]``, ``i = 0, 1``.\n\n    Parameters\n    ----------\n    old, new : array_like\n        Domains. Each domain must (successfully) convert to a 1-d array\n        containing precisely two values.\n\n    Returns\n    -------\n    offset, scale : scalars\n        The map ``L(x) = offset + scale*x`` maps the first domain to the\n        second.\n\n    See Also\n    --------\n    getdomain, mapdomain\n\n    Notes\n    -----\n    Also works for complex numbers, and thus can be used to calculate the\n    parameters required to map any line in the complex plane to any other\n    line therein.\n\n    Examples\n    --------\n    >>> from numpy import polynomial as P\n    >>> P.mapparms((-1,1),(-1,1))\n    (0.0, 1.0)\n    >>> P.mapparms((1,-1),(-1,1))\n    (0.0, -1.0)\n    >>> i = complex(0,1)\n    >>> P.mapparms((-i,-1),(1,i))\n    ((1+1j), (1+0j))\n\n    ')
    
    # Assigning a BinOp to a Name (line 332):
    
    # Assigning a BinOp to a Name (line 332):
    
    # Obtaining the type of the subscript
    int_179083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 17), 'int')
    # Getting the type of 'old' (line 332)
    old_179084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 13), 'old')
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___179085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 13), old_179084, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_179086 = invoke(stypy.reporting.localization.Localization(__file__, 332, 13), getitem___179085, int_179083)
    
    
    # Obtaining the type of the subscript
    int_179087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 26), 'int')
    # Getting the type of 'old' (line 332)
    old_179088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 22), 'old')
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___179089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 22), old_179088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_179090 = invoke(stypy.reporting.localization.Localization(__file__, 332, 22), getitem___179089, int_179087)
    
    # Applying the binary operator '-' (line 332)
    result_sub_179091 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 13), '-', subscript_call_result_179086, subscript_call_result_179090)
    
    # Assigning a type to the variable 'oldlen' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'oldlen', result_sub_179091)
    
    # Assigning a BinOp to a Name (line 333):
    
    # Assigning a BinOp to a Name (line 333):
    
    # Obtaining the type of the subscript
    int_179092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 17), 'int')
    # Getting the type of 'new' (line 333)
    new_179093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 13), 'new')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___179094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 13), new_179093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_179095 = invoke(stypy.reporting.localization.Localization(__file__, 333, 13), getitem___179094, int_179092)
    
    
    # Obtaining the type of the subscript
    int_179096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 26), 'int')
    # Getting the type of 'new' (line 333)
    new_179097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 22), 'new')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___179098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 22), new_179097, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_179099 = invoke(stypy.reporting.localization.Localization(__file__, 333, 22), getitem___179098, int_179096)
    
    # Applying the binary operator '-' (line 333)
    result_sub_179100 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 13), '-', subscript_call_result_179095, subscript_call_result_179099)
    
    # Assigning a type to the variable 'newlen' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'newlen', result_sub_179100)
    
    # Assigning a BinOp to a Name (line 334):
    
    # Assigning a BinOp to a Name (line 334):
    
    # Obtaining the type of the subscript
    int_179101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 15), 'int')
    # Getting the type of 'old' (line 334)
    old_179102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'old')
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___179103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 11), old_179102, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_179104 = invoke(stypy.reporting.localization.Localization(__file__, 334, 11), getitem___179103, int_179101)
    
    
    # Obtaining the type of the subscript
    int_179105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 22), 'int')
    # Getting the type of 'new' (line 334)
    new_179106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 18), 'new')
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___179107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 18), new_179106, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_179108 = invoke(stypy.reporting.localization.Localization(__file__, 334, 18), getitem___179107, int_179105)
    
    # Applying the binary operator '*' (line 334)
    result_mul_179109 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), '*', subscript_call_result_179104, subscript_call_result_179108)
    
    
    # Obtaining the type of the subscript
    int_179110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 31), 'int')
    # Getting the type of 'old' (line 334)
    old_179111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 27), 'old')
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___179112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 27), old_179111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_179113 = invoke(stypy.reporting.localization.Localization(__file__, 334, 27), getitem___179112, int_179110)
    
    
    # Obtaining the type of the subscript
    int_179114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 38), 'int')
    # Getting the type of 'new' (line 334)
    new_179115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 34), 'new')
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___179116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 34), new_179115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_179117 = invoke(stypy.reporting.localization.Localization(__file__, 334, 34), getitem___179116, int_179114)
    
    # Applying the binary operator '*' (line 334)
    result_mul_179118 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 27), '*', subscript_call_result_179113, subscript_call_result_179117)
    
    # Applying the binary operator '-' (line 334)
    result_sub_179119 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), '-', result_mul_179109, result_mul_179118)
    
    # Getting the type of 'oldlen' (line 334)
    oldlen_179120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 42), 'oldlen')
    # Applying the binary operator 'div' (line 334)
    result_div_179121 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 10), 'div', result_sub_179119, oldlen_179120)
    
    # Assigning a type to the variable 'off' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'off', result_div_179121)
    
    # Assigning a BinOp to a Name (line 335):
    
    # Assigning a BinOp to a Name (line 335):
    # Getting the type of 'newlen' (line 335)
    newlen_179122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 10), 'newlen')
    # Getting the type of 'oldlen' (line 335)
    oldlen_179123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'oldlen')
    # Applying the binary operator 'div' (line 335)
    result_div_179124 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 10), 'div', newlen_179122, oldlen_179123)
    
    # Assigning a type to the variable 'scl' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'scl', result_div_179124)
    
    # Obtaining an instance of the builtin type 'tuple' (line 336)
    tuple_179125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 336)
    # Adding element type (line 336)
    # Getting the type of 'off' (line 336)
    off_179126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'off')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 11), tuple_179125, off_179126)
    # Adding element type (line 336)
    # Getting the type of 'scl' (line 336)
    scl_179127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'scl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 11), tuple_179125, scl_179127)
    
    # Assigning a type to the variable 'stypy_return_type' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type', tuple_179125)
    
    # ################# End of 'mapparms(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mapparms' in the type store
    # Getting the type of 'stypy_return_type' (line 291)
    stypy_return_type_179128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_179128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mapparms'
    return stypy_return_type_179128

# Assigning a type to the variable 'mapparms' (line 291)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), 'mapparms', mapparms)

@norecursion
def mapdomain(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mapdomain'
    module_type_store = module_type_store.open_function_context('mapdomain', 338, 0, False)
    
    # Passed parameters checking function
    mapdomain.stypy_localization = localization
    mapdomain.stypy_type_of_self = None
    mapdomain.stypy_type_store = module_type_store
    mapdomain.stypy_function_name = 'mapdomain'
    mapdomain.stypy_param_names_list = ['x', 'old', 'new']
    mapdomain.stypy_varargs_param_name = None
    mapdomain.stypy_kwargs_param_name = None
    mapdomain.stypy_call_defaults = defaults
    mapdomain.stypy_call_varargs = varargs
    mapdomain.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mapdomain', ['x', 'old', 'new'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mapdomain', localization, ['x', 'old', 'new'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mapdomain(...)' code ##################

    str_179129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, (-1)), 'str', '\n    Apply linear map to input points.\n\n    The linear map ``offset + scale*x`` that maps the domain `old` to\n    the domain `new` is applied to the points `x`.\n\n    Parameters\n    ----------\n    x : array_like\n        Points to be mapped. If `x` is a subtype of ndarray the subtype\n        will be preserved.\n    old, new : array_like\n        The two domains that determine the map.  Each must (successfully)\n        convert to 1-d arrays containing precisely two values.\n\n    Returns\n    -------\n    x_out : ndarray\n        Array of points of the same shape as `x`, after application of the\n        linear map between the two domains.\n\n    See Also\n    --------\n    getdomain, mapparms\n\n    Notes\n    -----\n    Effectively, this implements:\n\n    .. math ::\n        x\\_out = new[0] + m(x - old[0])\n\n    where\n\n    .. math ::\n        m = \\frac{new[1]-new[0]}{old[1]-old[0]}\n\n    Examples\n    --------\n    >>> from numpy import polynomial as P\n    >>> old_domain = (-1,1)\n    >>> new_domain = (0,2*np.pi)\n    >>> x = np.linspace(-1,1,6); x\n    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ])\n    >>> x_out = P.mapdomain(x, old_domain, new_domain); x_out\n    array([ 0.        ,  1.25663706,  2.51327412,  3.76991118,  5.02654825,\n            6.28318531])\n    >>> x - P.mapdomain(x_out, new_domain, old_domain)\n    array([ 0.,  0.,  0.,  0.,  0.,  0.])\n\n    Also works for complex numbers (and thus can be used to map any line in\n    the complex plane to any other line therein).\n\n    >>> i = complex(0,1)\n    >>> old = (-1 - i, 1 + i)\n    >>> new = (-1 + i, 1 - i)\n    >>> z = np.linspace(old[0], old[1], 6); z\n    array([-1.0-1.j , -0.6-0.6j, -0.2-0.2j,  0.2+0.2j,  0.6+0.6j,  1.0+1.j ])\n    >>> new_z = P.mapdomain(z, old, new); new_z\n    array([-1.0+1.j , -0.6+0.6j, -0.2+0.2j,  0.2-0.2j,  0.6-0.6j,  1.0-1.j ])\n\n    ')
    
    # Assigning a Call to a Name (line 401):
    
    # Assigning a Call to a Name (line 401):
    
    # Call to asanyarray(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'x' (line 401)
    x_179132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 22), 'x', False)
    # Processing the call keyword arguments (line 401)
    kwargs_179133 = {}
    # Getting the type of 'np' (line 401)
    np_179130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 401)
    asanyarray_179131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), np_179130, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 401)
    asanyarray_call_result_179134 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), asanyarray_179131, *[x_179132], **kwargs_179133)
    
    # Assigning a type to the variable 'x' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'x', asanyarray_call_result_179134)
    
    # Assigning a Call to a Tuple (line 402):
    
    # Assigning a Call to a Name:
    
    # Call to mapparms(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'old' (line 402)
    old_179136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 24), 'old', False)
    # Getting the type of 'new' (line 402)
    new_179137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 29), 'new', False)
    # Processing the call keyword arguments (line 402)
    kwargs_179138 = {}
    # Getting the type of 'mapparms' (line 402)
    mapparms_179135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'mapparms', False)
    # Calling mapparms(args, kwargs) (line 402)
    mapparms_call_result_179139 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), mapparms_179135, *[old_179136, new_179137], **kwargs_179138)
    
    # Assigning a type to the variable 'call_assignment_178743' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'call_assignment_178743', mapparms_call_result_179139)
    
    # Assigning a Call to a Name (line 402):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_179142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 4), 'int')
    # Processing the call keyword arguments
    kwargs_179143 = {}
    # Getting the type of 'call_assignment_178743' (line 402)
    call_assignment_178743_179140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'call_assignment_178743', False)
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___179141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 4), call_assignment_178743_179140, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_179144 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179141, *[int_179142], **kwargs_179143)
    
    # Assigning a type to the variable 'call_assignment_178744' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'call_assignment_178744', getitem___call_result_179144)
    
    # Assigning a Name to a Name (line 402):
    # Getting the type of 'call_assignment_178744' (line 402)
    call_assignment_178744_179145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'call_assignment_178744')
    # Assigning a type to the variable 'off' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'off', call_assignment_178744_179145)
    
    # Assigning a Call to a Name (line 402):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_179148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 4), 'int')
    # Processing the call keyword arguments
    kwargs_179149 = {}
    # Getting the type of 'call_assignment_178743' (line 402)
    call_assignment_178743_179146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'call_assignment_178743', False)
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___179147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 4), call_assignment_178743_179146, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_179150 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179147, *[int_179148], **kwargs_179149)
    
    # Assigning a type to the variable 'call_assignment_178745' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'call_assignment_178745', getitem___call_result_179150)
    
    # Assigning a Name to a Name (line 402):
    # Getting the type of 'call_assignment_178745' (line 402)
    call_assignment_178745_179151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'call_assignment_178745')
    # Assigning a type to the variable 'scl' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 9), 'scl', call_assignment_178745_179151)
    # Getting the type of 'off' (line 403)
    off_179152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'off')
    # Getting the type of 'scl' (line 403)
    scl_179153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'scl')
    # Getting the type of 'x' (line 403)
    x_179154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 21), 'x')
    # Applying the binary operator '*' (line 403)
    result_mul_179155 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 17), '*', scl_179153, x_179154)
    
    # Applying the binary operator '+' (line 403)
    result_add_179156 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 11), '+', off_179152, result_mul_179155)
    
    # Assigning a type to the variable 'stypy_return_type' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type', result_add_179156)
    
    # ################# End of 'mapdomain(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mapdomain' in the type store
    # Getting the type of 'stypy_return_type' (line 338)
    stypy_return_type_179157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_179157)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mapdomain'
    return stypy_return_type_179157

# Assigning a type to the variable 'mapdomain' (line 338)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 0), 'mapdomain', mapdomain)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
