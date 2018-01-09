
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Machine arithmetics - determine the parameters of the
3: floating-point arithmetic system
4: 
5: Author: Pearu Peterson, September 2003
6: 
7: '''
8: from __future__ import division, absolute_import, print_function
9: 
10: __all__ = ['MachAr']
11: 
12: from numpy.core.fromnumeric import any
13: from numpy.core.numeric import errstate
14: 
15: # Need to speed this up...especially for longfloat
16: 
17: class MachAr(object):
18:     '''
19:     Diagnosing machine parameters.
20: 
21:     Attributes
22:     ----------
23:     ibeta : int
24:         Radix in which numbers are represented.
25:     it : int
26:         Number of base-`ibeta` digits in the floating point mantissa M.
27:     machep : int
28:         Exponent of the smallest (most negative) power of `ibeta` that,
29:         added to 1.0, gives something different from 1.0
30:     eps : float
31:         Floating-point number ``beta**machep`` (floating point precision)
32:     negep : int
33:         Exponent of the smallest power of `ibeta` that, substracted
34:         from 1.0, gives something different from 1.0.
35:     epsneg : float
36:         Floating-point number ``beta**negep``.
37:     iexp : int
38:         Number of bits in the exponent (including its sign and bias).
39:     minexp : int
40:         Smallest (most negative) power of `ibeta` consistent with there
41:         being no leading zeros in the mantissa.
42:     xmin : float
43:         Floating point number ``beta**minexp`` (the smallest [in
44:         magnitude] usable floating value).
45:     maxexp : int
46:         Smallest (positive) power of `ibeta` that causes overflow.
47:     xmax : float
48:         ``(1-epsneg) * beta**maxexp`` (the largest [in magnitude]
49:         usable floating value).
50:     irnd : int
51:         In ``range(6)``, information on what kind of rounding is done
52:         in addition, and on how underflow is handled.
53:     ngrd : int
54:         Number of 'guard digits' used when truncating the product
55:         of two mantissas to fit the representation.
56:     epsilon : float
57:         Same as `eps`.
58:     tiny : float
59:         Same as `xmin`.
60:     huge : float
61:         Same as `xmax`.
62:     precision : float
63:         ``- int(-log10(eps))``
64:     resolution : float
65:         ``- 10**(-precision)``
66: 
67:     Parameters
68:     ----------
69:     float_conv : function, optional
70:         Function that converts an integer or integer array to a float
71:         or float array. Default is `float`.
72:     int_conv : function, optional
73:         Function that converts a float or float array to an integer or
74:         integer array. Default is `int`.
75:     float_to_float : function, optional
76:         Function that converts a float array to float. Default is `float`.
77:         Note that this does not seem to do anything useful in the current
78:         implementation.
79:     float_to_str : function, optional
80:         Function that converts a single float to a string. Default is
81:         ``lambda v:'%24.16e' %v``.
82:     title : str, optional
83:         Title that is printed in the string representation of `MachAr`.
84: 
85:     See Also
86:     --------
87:     finfo : Machine limits for floating point types.
88:     iinfo : Machine limits for integer types.
89: 
90:     References
91:     ----------
92:     .. [1] Press, Teukolsky, Vetterling and Flannery,
93:            "Numerical Recipes in C++," 2nd ed,
94:            Cambridge University Press, 2002, p. 31.
95: 
96:     '''
97: 
98:     def __init__(self, float_conv=float,int_conv=int,
99:                  float_to_float=float,
100:                  float_to_str=lambda v:'%24.16e' % v,
101:                  title='Python floating point number'):
102:         '''
103: 
104:         float_conv - convert integer to float (array)
105:         int_conv   - convert float (array) to integer
106:         float_to_float - convert float array to float
107:         float_to_str - convert array float to str
108:         title        - description of used floating point numbers
109: 
110:         '''
111:         # We ignore all errors here because we are purposely triggering
112:         # underflow to detect the properties of the runninng arch.
113:         with errstate(under='ignore'):
114:             self._do_init(float_conv, int_conv, float_to_float, float_to_str, title)
115: 
116:     def _do_init(self, float_conv, int_conv, float_to_float, float_to_str, title):
117:         max_iterN = 10000
118:         msg = "Did not converge after %d tries with %s"
119:         one = float_conv(1)
120:         two = one + one
121:         zero = one - one
122: 
123:         # Do we really need to do this?  Aren't they 2 and 2.0?
124:         # Determine ibeta and beta
125:         a = one
126:         for _ in range(max_iterN):
127:             a = a + a
128:             temp = a + one
129:             temp1 = temp - a
130:             if any(temp1 - one != zero):
131:                 break
132:         else:
133:             raise RuntimeError(msg % (_, one.dtype))
134:         b = one
135:         for _ in range(max_iterN):
136:             b = b + b
137:             temp = a + b
138:             itemp = int_conv(temp-a)
139:             if any(itemp != 0):
140:                 break
141:         else:
142:             raise RuntimeError(msg % (_, one.dtype))
143:         ibeta = itemp
144:         beta = float_conv(ibeta)
145: 
146:         # Determine it and irnd
147:         it = -1
148:         b = one
149:         for _ in range(max_iterN):
150:             it = it + 1
151:             b = b * beta
152:             temp = b + one
153:             temp1 = temp - b
154:             if any(temp1 - one != zero):
155:                 break
156:         else:
157:             raise RuntimeError(msg % (_, one.dtype))
158: 
159:         betah = beta / two
160:         a = one
161:         for _ in range(max_iterN):
162:             a = a + a
163:             temp = a + one
164:             temp1 = temp - a
165:             if any(temp1 - one != zero):
166:                 break
167:         else:
168:             raise RuntimeError(msg % (_, one.dtype))
169:         temp = a + betah
170:         irnd = 0
171:         if any(temp-a != zero):
172:             irnd = 1
173:         tempa = a + beta
174:         temp = tempa + betah
175:         if irnd == 0 and any(temp-tempa != zero):
176:             irnd = 2
177: 
178:         # Determine negep and epsneg
179:         negep = it + 3
180:         betain = one / beta
181:         a = one
182:         for i in range(negep):
183:             a = a * betain
184:         b = a
185:         for _ in range(max_iterN):
186:             temp = one - a
187:             if any(temp-one != zero):
188:                 break
189:             a = a * beta
190:             negep = negep - 1
191:             # Prevent infinite loop on PPC with gcc 4.0:
192:             if negep < 0:
193:                 raise RuntimeError("could not determine machine tolerance "
194:                                    "for 'negep', locals() -> %s" % (locals()))
195:         else:
196:             raise RuntimeError(msg % (_, one.dtype))
197:         negep = -negep
198:         epsneg = a
199: 
200:         # Determine machep and eps
201:         machep = - it - 3
202:         a = b
203: 
204:         for _ in range(max_iterN):
205:             temp = one + a
206:             if any(temp-one != zero):
207:                 break
208:             a = a * beta
209:             machep = machep + 1
210:         else:
211:             raise RuntimeError(msg % (_, one.dtype))
212:         eps = a
213: 
214:         # Determine ngrd
215:         ngrd = 0
216:         temp = one + eps
217:         if irnd == 0 and any(temp*one - one != zero):
218:             ngrd = 1
219: 
220:         # Determine iexp
221:         i = 0
222:         k = 1
223:         z = betain
224:         t = one + eps
225:         nxres = 0
226:         for _ in range(max_iterN):
227:             y = z
228:             z = y*y
229:             a = z*one  # Check here for underflow
230:             temp = z*t
231:             if any(a+a == zero) or any(abs(z) >= y):
232:                 break
233:             temp1 = temp * betain
234:             if any(temp1*beta == z):
235:                 break
236:             i = i + 1
237:             k = k + k
238:         else:
239:             raise RuntimeError(msg % (_, one.dtype))
240:         if ibeta != 10:
241:             iexp = i + 1
242:             mx = k + k
243:         else:
244:             iexp = 2
245:             iz = ibeta
246:             while k >= iz:
247:                 iz = iz * ibeta
248:                 iexp = iexp + 1
249:             mx = iz + iz - 1
250: 
251:         # Determine minexp and xmin
252:         for _ in range(max_iterN):
253:             xmin = y
254:             y = y * betain
255:             a = y * one
256:             temp = y * t
257:             if any((a + a) != zero) and any(abs(y) < xmin):
258:                 k = k + 1
259:                 temp1 = temp * betain
260:                 if any(temp1*beta == y) and any(temp != y):
261:                     nxres = 3
262:                     xmin = y
263:                     break
264:             else:
265:                 break
266:         else:
267:             raise RuntimeError(msg % (_, one.dtype))
268:         minexp = -k
269: 
270:         # Determine maxexp, xmax
271:         if mx <= k + k - 3 and ibeta != 10:
272:             mx = mx + mx
273:             iexp = iexp + 1
274:         maxexp = mx + minexp
275:         irnd = irnd + nxres
276:         if irnd >= 2:
277:             maxexp = maxexp - 2
278:         i = maxexp + minexp
279:         if ibeta == 2 and not i:
280:             maxexp = maxexp - 1
281:         if i > 20:
282:             maxexp = maxexp - 1
283:         if any(a != y):
284:             maxexp = maxexp - 2
285:         xmax = one - epsneg
286:         if any(xmax*one != xmax):
287:             xmax = one - beta*epsneg
288:         xmax = xmax / (xmin*beta*beta*beta)
289:         i = maxexp + minexp + 3
290:         for j in range(i):
291:             if ibeta == 2:
292:                 xmax = xmax + xmax
293:             else:
294:                 xmax = xmax * beta
295: 
296:         self.ibeta = ibeta
297:         self.it = it
298:         self.negep = negep
299:         self.epsneg = float_to_float(epsneg)
300:         self._str_epsneg = float_to_str(epsneg)
301:         self.machep = machep
302:         self.eps = float_to_float(eps)
303:         self._str_eps = float_to_str(eps)
304:         self.ngrd = ngrd
305:         self.iexp = iexp
306:         self.minexp = minexp
307:         self.xmin = float_to_float(xmin)
308:         self._str_xmin = float_to_str(xmin)
309:         self.maxexp = maxexp
310:         self.xmax = float_to_float(xmax)
311:         self._str_xmax = float_to_str(xmax)
312:         self.irnd = irnd
313: 
314:         self.title = title
315:         # Commonly used parameters
316:         self.epsilon = self.eps
317:         self.tiny = self.xmin
318:         self.huge = self.xmax
319: 
320:         import math
321:         self.precision = int(-math.log10(float_to_float(self.eps)))
322:         ten = two + two + two + two + two
323:         resolution = ten ** (-self.precision)
324:         self.resolution = float_to_float(resolution)
325:         self._str_resolution = float_to_str(resolution)
326: 
327:     def __str__(self):
328:         fmt = (
329:            'Machine parameters for %(title)s\n'
330:            '---------------------------------------------------------------------\n'
331:            'ibeta=%(ibeta)s it=%(it)s iexp=%(iexp)s ngrd=%(ngrd)s irnd=%(irnd)s\n'
332:            'machep=%(machep)s     eps=%(_str_eps)s (beta**machep == epsilon)\n'
333:            'negep =%(negep)s  epsneg=%(_str_epsneg)s (beta**epsneg)\n'
334:            'minexp=%(minexp)s   xmin=%(_str_xmin)s (beta**minexp == tiny)\n'
335:            'maxexp=%(maxexp)s    xmax=%(_str_xmax)s ((1-epsneg)*beta**maxexp == huge)\n'
336:            '---------------------------------------------------------------------\n'
337:            )
338:         return fmt % self.__dict__
339: 
340: 
341: if __name__ == '__main__':
342:     print(MachAr())
343: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_6353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nMachine arithmetics - determine the parameters of the\nfloating-point arithmetic system\n\nAuthor: Pearu Peterson, September 2003\n\n')

# Assigning a List to a Name (line 10):
__all__ = ['MachAr']
module_type_store.set_exportable_members(['MachAr'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_6354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_6355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'MachAr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_6354, str_6355)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_6354)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.core.fromnumeric import any' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_6356 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.fromnumeric')

if (type(import_6356) is not StypyTypeError):

    if (import_6356 != 'pyd_module'):
        __import__(import_6356)
        sys_modules_6357 = sys.modules[import_6356]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.fromnumeric', sys_modules_6357.module_type_store, module_type_store, ['any'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_6357, sys_modules_6357.module_type_store, module_type_store)
    else:
        from numpy.core.fromnumeric import any

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.fromnumeric', None, module_type_store, ['any'], [any])

else:
    # Assigning a type to the variable 'numpy.core.fromnumeric' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.fromnumeric', import_6356)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.core.numeric import errstate' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_6358 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.numeric')

if (type(import_6358) is not StypyTypeError):

    if (import_6358 != 'pyd_module'):
        __import__(import_6358)
        sys_modules_6359 = sys.modules[import_6358]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.numeric', sys_modules_6359.module_type_store, module_type_store, ['errstate'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_6359, sys_modules_6359.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import errstate

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.numeric', None, module_type_store, ['errstate'], [errstate])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.numeric', import_6358)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

# Declaration of the 'MachAr' class

class MachAr(object, ):
    str_6360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, (-1)), 'str', '\n    Diagnosing machine parameters.\n\n    Attributes\n    ----------\n    ibeta : int\n        Radix in which numbers are represented.\n    it : int\n        Number of base-`ibeta` digits in the floating point mantissa M.\n    machep : int\n        Exponent of the smallest (most negative) power of `ibeta` that,\n        added to 1.0, gives something different from 1.0\n    eps : float\n        Floating-point number ``beta**machep`` (floating point precision)\n    negep : int\n        Exponent of the smallest power of `ibeta` that, substracted\n        from 1.0, gives something different from 1.0.\n    epsneg : float\n        Floating-point number ``beta**negep``.\n    iexp : int\n        Number of bits in the exponent (including its sign and bias).\n    minexp : int\n        Smallest (most negative) power of `ibeta` consistent with there\n        being no leading zeros in the mantissa.\n    xmin : float\n        Floating point number ``beta**minexp`` (the smallest [in\n        magnitude] usable floating value).\n    maxexp : int\n        Smallest (positive) power of `ibeta` that causes overflow.\n    xmax : float\n        ``(1-epsneg) * beta**maxexp`` (the largest [in magnitude]\n        usable floating value).\n    irnd : int\n        In ``range(6)``, information on what kind of rounding is done\n        in addition, and on how underflow is handled.\n    ngrd : int\n        Number of \'guard digits\' used when truncating the product\n        of two mantissas to fit the representation.\n    epsilon : float\n        Same as `eps`.\n    tiny : float\n        Same as `xmin`.\n    huge : float\n        Same as `xmax`.\n    precision : float\n        ``- int(-log10(eps))``\n    resolution : float\n        ``- 10**(-precision)``\n\n    Parameters\n    ----------\n    float_conv : function, optional\n        Function that converts an integer or integer array to a float\n        or float array. Default is `float`.\n    int_conv : function, optional\n        Function that converts a float or float array to an integer or\n        integer array. Default is `int`.\n    float_to_float : function, optional\n        Function that converts a float array to float. Default is `float`.\n        Note that this does not seem to do anything useful in the current\n        implementation.\n    float_to_str : function, optional\n        Function that converts a single float to a string. Default is\n        ``lambda v:\'%24.16e\' %v``.\n    title : str, optional\n        Title that is printed in the string representation of `MachAr`.\n\n    See Also\n    --------\n    finfo : Machine limits for floating point types.\n    iinfo : Machine limits for integer types.\n\n    References\n    ----------\n    .. [1] Press, Teukolsky, Vetterling and Flannery,\n           "Numerical Recipes in C++," 2nd ed,\n           Cambridge University Press, 2002, p. 31.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'float' (line 98)
        float_6361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'float')
        # Getting the type of 'int' (line 98)
        int_6362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 49), 'int')
        # Getting the type of 'float' (line 99)
        float_6363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 32), 'float')

        @norecursion
        def _stypy_temp_lambda_5(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_5'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_5', 100, 30, True)
            # Passed parameters checking function
            _stypy_temp_lambda_5.stypy_localization = localization
            _stypy_temp_lambda_5.stypy_type_of_self = None
            _stypy_temp_lambda_5.stypy_type_store = module_type_store
            _stypy_temp_lambda_5.stypy_function_name = '_stypy_temp_lambda_5'
            _stypy_temp_lambda_5.stypy_param_names_list = ['v']
            _stypy_temp_lambda_5.stypy_varargs_param_name = None
            _stypy_temp_lambda_5.stypy_kwargs_param_name = None
            _stypy_temp_lambda_5.stypy_call_defaults = defaults
            _stypy_temp_lambda_5.stypy_call_varargs = varargs
            _stypy_temp_lambda_5.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_5', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_5', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            str_6364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 39), 'str', '%24.16e')
            # Getting the type of 'v' (line 100)
            v_6365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 51), 'v')
            # Applying the binary operator '%' (line 100)
            result_mod_6366 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 39), '%', str_6364, v_6365)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'stypy_return_type', result_mod_6366)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_5' in the type store
            # Getting the type of 'stypy_return_type' (line 100)
            stypy_return_type_6367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6367)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_5'
            return stypy_return_type_6367

        # Assigning a type to the variable '_stypy_temp_lambda_5' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), '_stypy_temp_lambda_5', _stypy_temp_lambda_5)
        # Getting the type of '_stypy_temp_lambda_5' (line 100)
        _stypy_temp_lambda_5_6368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), '_stypy_temp_lambda_5')
        str_6369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'str', 'Python floating point number')
        defaults = [float_6361, int_6362, float_6363, _stypy_temp_lambda_5_6368, str_6369]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MachAr.__init__', ['float_conv', 'int_conv', 'float_to_float', 'float_to_str', 'title'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['float_conv', 'int_conv', 'float_to_float', 'float_to_str', 'title'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_6370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', '\n\n        float_conv - convert integer to float (array)\n        int_conv   - convert float (array) to integer\n        float_to_float - convert float array to float\n        float_to_str - convert array float to str\n        title        - description of used floating point numbers\n\n        ')
        
        # Call to errstate(...): (line 113)
        # Processing the call keyword arguments (line 113)
        str_6372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'str', 'ignore')
        keyword_6373 = str_6372
        kwargs_6374 = {'under': keyword_6373}
        # Getting the type of 'errstate' (line 113)
        errstate_6371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'errstate', False)
        # Calling errstate(args, kwargs) (line 113)
        errstate_call_result_6375 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), errstate_6371, *[], **kwargs_6374)
        
        with_6376 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 113, 13), errstate_call_result_6375, 'with parameter', '__enter__', '__exit__')

        if with_6376:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 113)
            enter___6377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 13), errstate_call_result_6375, '__enter__')
            with_enter_6378 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), enter___6377)
            
            # Call to _do_init(...): (line 114)
            # Processing the call arguments (line 114)
            # Getting the type of 'float_conv' (line 114)
            float_conv_6381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'float_conv', False)
            # Getting the type of 'int_conv' (line 114)
            int_conv_6382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'int_conv', False)
            # Getting the type of 'float_to_float' (line 114)
            float_to_float_6383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 48), 'float_to_float', False)
            # Getting the type of 'float_to_str' (line 114)
            float_to_str_6384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 64), 'float_to_str', False)
            # Getting the type of 'title' (line 114)
            title_6385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 78), 'title', False)
            # Processing the call keyword arguments (line 114)
            kwargs_6386 = {}
            # Getting the type of 'self' (line 114)
            self_6379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'self', False)
            # Obtaining the member '_do_init' of a type (line 114)
            _do_init_6380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), self_6379, '_do_init')
            # Calling _do_init(args, kwargs) (line 114)
            _do_init_call_result_6387 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), _do_init_6380, *[float_conv_6381, int_conv_6382, float_to_float_6383, float_to_str_6384, title_6385], **kwargs_6386)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 113)
            exit___6388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 13), errstate_call_result_6375, '__exit__')
            with_exit_6389 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), exit___6388, None, None, None)

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _do_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_do_init'
        module_type_store = module_type_store.open_function_context('_do_init', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MachAr._do_init.__dict__.__setitem__('stypy_localization', localization)
        MachAr._do_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MachAr._do_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        MachAr._do_init.__dict__.__setitem__('stypy_function_name', 'MachAr._do_init')
        MachAr._do_init.__dict__.__setitem__('stypy_param_names_list', ['float_conv', 'int_conv', 'float_to_float', 'float_to_str', 'title'])
        MachAr._do_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        MachAr._do_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MachAr._do_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        MachAr._do_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        MachAr._do_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MachAr._do_init.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MachAr._do_init', ['float_conv', 'int_conv', 'float_to_float', 'float_to_str', 'title'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_do_init', localization, ['float_conv', 'int_conv', 'float_to_float', 'float_to_str', 'title'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_do_init(...)' code ##################

        
        # Assigning a Num to a Name (line 117):
        int_6390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'int')
        # Assigning a type to the variable 'max_iterN' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'max_iterN', int_6390)
        
        # Assigning a Str to a Name (line 118):
        str_6391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 14), 'str', 'Did not converge after %d tries with %s')
        # Assigning a type to the variable 'msg' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'msg', str_6391)
        
        # Assigning a Call to a Name (line 119):
        
        # Call to float_conv(...): (line 119)
        # Processing the call arguments (line 119)
        int_6393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 25), 'int')
        # Processing the call keyword arguments (line 119)
        kwargs_6394 = {}
        # Getting the type of 'float_conv' (line 119)
        float_conv_6392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'float_conv', False)
        # Calling float_conv(args, kwargs) (line 119)
        float_conv_call_result_6395 = invoke(stypy.reporting.localization.Localization(__file__, 119, 14), float_conv_6392, *[int_6393], **kwargs_6394)
        
        # Assigning a type to the variable 'one' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'one', float_conv_call_result_6395)
        
        # Assigning a BinOp to a Name (line 120):
        # Getting the type of 'one' (line 120)
        one_6396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'one')
        # Getting the type of 'one' (line 120)
        one_6397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'one')
        # Applying the binary operator '+' (line 120)
        result_add_6398 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 14), '+', one_6396, one_6397)
        
        # Assigning a type to the variable 'two' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'two', result_add_6398)
        
        # Assigning a BinOp to a Name (line 121):
        # Getting the type of 'one' (line 121)
        one_6399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'one')
        # Getting the type of 'one' (line 121)
        one_6400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'one')
        # Applying the binary operator '-' (line 121)
        result_sub_6401 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 15), '-', one_6399, one_6400)
        
        # Assigning a type to the variable 'zero' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'zero', result_sub_6401)
        
        # Assigning a Name to a Name (line 125):
        # Getting the type of 'one' (line 125)
        one_6402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'one')
        # Assigning a type to the variable 'a' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'a', one_6402)
        
        
        # Call to range(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'max_iterN' (line 126)
        max_iterN_6404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'max_iterN', False)
        # Processing the call keyword arguments (line 126)
        kwargs_6405 = {}
        # Getting the type of 'range' (line 126)
        range_6403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'range', False)
        # Calling range(args, kwargs) (line 126)
        range_call_result_6406 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), range_6403, *[max_iterN_6404], **kwargs_6405)
        
        # Testing the type of a for loop iterable (line 126)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 8), range_call_result_6406)
        # Getting the type of the for loop variable (line 126)
        for_loop_var_6407 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 8), range_call_result_6406)
        # Assigning a type to the variable '_' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), '_', for_loop_var_6407)
        # SSA begins for a for statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 127):
        # Getting the type of 'a' (line 127)
        a_6408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'a')
        # Getting the type of 'a' (line 127)
        a_6409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'a')
        # Applying the binary operator '+' (line 127)
        result_add_6410 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 16), '+', a_6408, a_6409)
        
        # Assigning a type to the variable 'a' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'a', result_add_6410)
        
        # Assigning a BinOp to a Name (line 128):
        # Getting the type of 'a' (line 128)
        a_6411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'a')
        # Getting the type of 'one' (line 128)
        one_6412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'one')
        # Applying the binary operator '+' (line 128)
        result_add_6413 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 19), '+', a_6411, one_6412)
        
        # Assigning a type to the variable 'temp' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'temp', result_add_6413)
        
        # Assigning a BinOp to a Name (line 129):
        # Getting the type of 'temp' (line 129)
        temp_6414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'temp')
        # Getting the type of 'a' (line 129)
        a_6415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'a')
        # Applying the binary operator '-' (line 129)
        result_sub_6416 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 20), '-', temp_6414, a_6415)
        
        # Assigning a type to the variable 'temp1' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'temp1', result_sub_6416)
        
        
        # Call to any(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Getting the type of 'temp1' (line 130)
        temp1_6418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 'temp1', False)
        # Getting the type of 'one' (line 130)
        one_6419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'one', False)
        # Applying the binary operator '-' (line 130)
        result_sub_6420 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 19), '-', temp1_6418, one_6419)
        
        # Getting the type of 'zero' (line 130)
        zero_6421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 34), 'zero', False)
        # Applying the binary operator '!=' (line 130)
        result_ne_6422 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 19), '!=', result_sub_6420, zero_6421)
        
        # Processing the call keyword arguments (line 130)
        kwargs_6423 = {}
        # Getting the type of 'any' (line 130)
        any_6417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'any', False)
        # Calling any(args, kwargs) (line 130)
        any_call_result_6424 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), any_6417, *[result_ne_6422], **kwargs_6423)
        
        # Testing the type of an if condition (line 130)
        if_condition_6425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 12), any_call_result_6424)
        # Assigning a type to the variable 'if_condition_6425' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'if_condition_6425', if_condition_6425)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 126)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to RuntimeError(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'msg' (line 133)
        msg_6427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 133)
        tuple_6428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 133)
        # Adding element type (line 133)
        # Getting the type of '_' (line 133)
        __6429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), '_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 38), tuple_6428, __6429)
        # Adding element type (line 133)
        # Getting the type of 'one' (line 133)
        one_6430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), 'one', False)
        # Obtaining the member 'dtype' of a type (line 133)
        dtype_6431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 41), one_6430, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 38), tuple_6428, dtype_6431)
        
        # Applying the binary operator '%' (line 133)
        result_mod_6432 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 31), '%', msg_6427, tuple_6428)
        
        # Processing the call keyword arguments (line 133)
        kwargs_6433 = {}
        # Getting the type of 'RuntimeError' (line 133)
        RuntimeError_6426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 133)
        RuntimeError_call_result_6434 = invoke(stypy.reporting.localization.Localization(__file__, 133, 18), RuntimeError_6426, *[result_mod_6432], **kwargs_6433)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 133, 12), RuntimeError_call_result_6434, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 134):
        # Getting the type of 'one' (line 134)
        one_6435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'one')
        # Assigning a type to the variable 'b' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'b', one_6435)
        
        
        # Call to range(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'max_iterN' (line 135)
        max_iterN_6437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'max_iterN', False)
        # Processing the call keyword arguments (line 135)
        kwargs_6438 = {}
        # Getting the type of 'range' (line 135)
        range_6436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'range', False)
        # Calling range(args, kwargs) (line 135)
        range_call_result_6439 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), range_6436, *[max_iterN_6437], **kwargs_6438)
        
        # Testing the type of a for loop iterable (line 135)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_6439)
        # Getting the type of the for loop variable (line 135)
        for_loop_var_6440 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_6439)
        # Assigning a type to the variable '_' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), '_', for_loop_var_6440)
        # SSA begins for a for statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 136):
        # Getting the type of 'b' (line 136)
        b_6441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'b')
        # Getting the type of 'b' (line 136)
        b_6442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'b')
        # Applying the binary operator '+' (line 136)
        result_add_6443 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 16), '+', b_6441, b_6442)
        
        # Assigning a type to the variable 'b' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'b', result_add_6443)
        
        # Assigning a BinOp to a Name (line 137):
        # Getting the type of 'a' (line 137)
        a_6444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'a')
        # Getting the type of 'b' (line 137)
        b_6445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'b')
        # Applying the binary operator '+' (line 137)
        result_add_6446 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 19), '+', a_6444, b_6445)
        
        # Assigning a type to the variable 'temp' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'temp', result_add_6446)
        
        # Assigning a Call to a Name (line 138):
        
        # Call to int_conv(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'temp' (line 138)
        temp_6448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'temp', False)
        # Getting the type of 'a' (line 138)
        a_6449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'a', False)
        # Applying the binary operator '-' (line 138)
        result_sub_6450 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 29), '-', temp_6448, a_6449)
        
        # Processing the call keyword arguments (line 138)
        kwargs_6451 = {}
        # Getting the type of 'int_conv' (line 138)
        int_conv_6447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'int_conv', False)
        # Calling int_conv(args, kwargs) (line 138)
        int_conv_call_result_6452 = invoke(stypy.reporting.localization.Localization(__file__, 138, 20), int_conv_6447, *[result_sub_6450], **kwargs_6451)
        
        # Assigning a type to the variable 'itemp' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'itemp', int_conv_call_result_6452)
        
        
        # Call to any(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Getting the type of 'itemp' (line 139)
        itemp_6454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'itemp', False)
        int_6455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 28), 'int')
        # Applying the binary operator '!=' (line 139)
        result_ne_6456 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 19), '!=', itemp_6454, int_6455)
        
        # Processing the call keyword arguments (line 139)
        kwargs_6457 = {}
        # Getting the type of 'any' (line 139)
        any_6453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'any', False)
        # Calling any(args, kwargs) (line 139)
        any_call_result_6458 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), any_6453, *[result_ne_6456], **kwargs_6457)
        
        # Testing the type of an if condition (line 139)
        if_condition_6459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 12), any_call_result_6458)
        # Assigning a type to the variable 'if_condition_6459' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'if_condition_6459', if_condition_6459)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 135)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to RuntimeError(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'msg' (line 142)
        msg_6461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_6462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of '_' (line 142)
        __6463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), '_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 38), tuple_6462, __6463)
        # Adding element type (line 142)
        # Getting the type of 'one' (line 142)
        one_6464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 41), 'one', False)
        # Obtaining the member 'dtype' of a type (line 142)
        dtype_6465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 41), one_6464, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 38), tuple_6462, dtype_6465)
        
        # Applying the binary operator '%' (line 142)
        result_mod_6466 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 31), '%', msg_6461, tuple_6462)
        
        # Processing the call keyword arguments (line 142)
        kwargs_6467 = {}
        # Getting the type of 'RuntimeError' (line 142)
        RuntimeError_6460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 142)
        RuntimeError_call_result_6468 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), RuntimeError_6460, *[result_mod_6466], **kwargs_6467)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 142, 12), RuntimeError_call_result_6468, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 143):
        # Getting the type of 'itemp' (line 143)
        itemp_6469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'itemp')
        # Assigning a type to the variable 'ibeta' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'ibeta', itemp_6469)
        
        # Assigning a Call to a Name (line 144):
        
        # Call to float_conv(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'ibeta' (line 144)
        ibeta_6471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'ibeta', False)
        # Processing the call keyword arguments (line 144)
        kwargs_6472 = {}
        # Getting the type of 'float_conv' (line 144)
        float_conv_6470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'float_conv', False)
        # Calling float_conv(args, kwargs) (line 144)
        float_conv_call_result_6473 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), float_conv_6470, *[ibeta_6471], **kwargs_6472)
        
        # Assigning a type to the variable 'beta' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'beta', float_conv_call_result_6473)
        
        # Assigning a Num to a Name (line 147):
        int_6474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 13), 'int')
        # Assigning a type to the variable 'it' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'it', int_6474)
        
        # Assigning a Name to a Name (line 148):
        # Getting the type of 'one' (line 148)
        one_6475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'one')
        # Assigning a type to the variable 'b' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'b', one_6475)
        
        
        # Call to range(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'max_iterN' (line 149)
        max_iterN_6477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'max_iterN', False)
        # Processing the call keyword arguments (line 149)
        kwargs_6478 = {}
        # Getting the type of 'range' (line 149)
        range_6476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'range', False)
        # Calling range(args, kwargs) (line 149)
        range_call_result_6479 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), range_6476, *[max_iterN_6477], **kwargs_6478)
        
        # Testing the type of a for loop iterable (line 149)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 149, 8), range_call_result_6479)
        # Getting the type of the for loop variable (line 149)
        for_loop_var_6480 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 149, 8), range_call_result_6479)
        # Assigning a type to the variable '_' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), '_', for_loop_var_6480)
        # SSA begins for a for statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 150):
        # Getting the type of 'it' (line 150)
        it_6481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'it')
        int_6482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 22), 'int')
        # Applying the binary operator '+' (line 150)
        result_add_6483 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 17), '+', it_6481, int_6482)
        
        # Assigning a type to the variable 'it' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'it', result_add_6483)
        
        # Assigning a BinOp to a Name (line 151):
        # Getting the type of 'b' (line 151)
        b_6484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'b')
        # Getting the type of 'beta' (line 151)
        beta_6485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'beta')
        # Applying the binary operator '*' (line 151)
        result_mul_6486 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 16), '*', b_6484, beta_6485)
        
        # Assigning a type to the variable 'b' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'b', result_mul_6486)
        
        # Assigning a BinOp to a Name (line 152):
        # Getting the type of 'b' (line 152)
        b_6487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'b')
        # Getting the type of 'one' (line 152)
        one_6488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'one')
        # Applying the binary operator '+' (line 152)
        result_add_6489 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 19), '+', b_6487, one_6488)
        
        # Assigning a type to the variable 'temp' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'temp', result_add_6489)
        
        # Assigning a BinOp to a Name (line 153):
        # Getting the type of 'temp' (line 153)
        temp_6490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'temp')
        # Getting the type of 'b' (line 153)
        b_6491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 27), 'b')
        # Applying the binary operator '-' (line 153)
        result_sub_6492 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 20), '-', temp_6490, b_6491)
        
        # Assigning a type to the variable 'temp1' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'temp1', result_sub_6492)
        
        
        # Call to any(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Getting the type of 'temp1' (line 154)
        temp1_6494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'temp1', False)
        # Getting the type of 'one' (line 154)
        one_6495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'one', False)
        # Applying the binary operator '-' (line 154)
        result_sub_6496 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '-', temp1_6494, one_6495)
        
        # Getting the type of 'zero' (line 154)
        zero_6497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'zero', False)
        # Applying the binary operator '!=' (line 154)
        result_ne_6498 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '!=', result_sub_6496, zero_6497)
        
        # Processing the call keyword arguments (line 154)
        kwargs_6499 = {}
        # Getting the type of 'any' (line 154)
        any_6493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'any', False)
        # Calling any(args, kwargs) (line 154)
        any_call_result_6500 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), any_6493, *[result_ne_6498], **kwargs_6499)
        
        # Testing the type of an if condition (line 154)
        if_condition_6501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 12), any_call_result_6500)
        # Assigning a type to the variable 'if_condition_6501' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'if_condition_6501', if_condition_6501)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 149)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to RuntimeError(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'msg' (line 157)
        msg_6503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 157)
        tuple_6504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 157)
        # Adding element type (line 157)
        # Getting the type of '_' (line 157)
        __6505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 38), '_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 38), tuple_6504, __6505)
        # Adding element type (line 157)
        # Getting the type of 'one' (line 157)
        one_6506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 41), 'one', False)
        # Obtaining the member 'dtype' of a type (line 157)
        dtype_6507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 41), one_6506, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 38), tuple_6504, dtype_6507)
        
        # Applying the binary operator '%' (line 157)
        result_mod_6508 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 31), '%', msg_6503, tuple_6504)
        
        # Processing the call keyword arguments (line 157)
        kwargs_6509 = {}
        # Getting the type of 'RuntimeError' (line 157)
        RuntimeError_6502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 157)
        RuntimeError_call_result_6510 = invoke(stypy.reporting.localization.Localization(__file__, 157, 18), RuntimeError_6502, *[result_mod_6508], **kwargs_6509)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 157, 12), RuntimeError_call_result_6510, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 159):
        # Getting the type of 'beta' (line 159)
        beta_6511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'beta')
        # Getting the type of 'two' (line 159)
        two_6512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'two')
        # Applying the binary operator 'div' (line 159)
        result_div_6513 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 16), 'div', beta_6511, two_6512)
        
        # Assigning a type to the variable 'betah' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'betah', result_div_6513)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'one' (line 160)
        one_6514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'one')
        # Assigning a type to the variable 'a' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'a', one_6514)
        
        
        # Call to range(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'max_iterN' (line 161)
        max_iterN_6516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'max_iterN', False)
        # Processing the call keyword arguments (line 161)
        kwargs_6517 = {}
        # Getting the type of 'range' (line 161)
        range_6515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'range', False)
        # Calling range(args, kwargs) (line 161)
        range_call_result_6518 = invoke(stypy.reporting.localization.Localization(__file__, 161, 17), range_6515, *[max_iterN_6516], **kwargs_6517)
        
        # Testing the type of a for loop iterable (line 161)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 161, 8), range_call_result_6518)
        # Getting the type of the for loop variable (line 161)
        for_loop_var_6519 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 161, 8), range_call_result_6518)
        # Assigning a type to the variable '_' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), '_', for_loop_var_6519)
        # SSA begins for a for statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 162):
        # Getting the type of 'a' (line 162)
        a_6520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'a')
        # Getting the type of 'a' (line 162)
        a_6521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'a')
        # Applying the binary operator '+' (line 162)
        result_add_6522 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 16), '+', a_6520, a_6521)
        
        # Assigning a type to the variable 'a' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'a', result_add_6522)
        
        # Assigning a BinOp to a Name (line 163):
        # Getting the type of 'a' (line 163)
        a_6523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'a')
        # Getting the type of 'one' (line 163)
        one_6524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'one')
        # Applying the binary operator '+' (line 163)
        result_add_6525 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 19), '+', a_6523, one_6524)
        
        # Assigning a type to the variable 'temp' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'temp', result_add_6525)
        
        # Assigning a BinOp to a Name (line 164):
        # Getting the type of 'temp' (line 164)
        temp_6526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'temp')
        # Getting the type of 'a' (line 164)
        a_6527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'a')
        # Applying the binary operator '-' (line 164)
        result_sub_6528 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 20), '-', temp_6526, a_6527)
        
        # Assigning a type to the variable 'temp1' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'temp1', result_sub_6528)
        
        
        # Call to any(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Getting the type of 'temp1' (line 165)
        temp1_6530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'temp1', False)
        # Getting the type of 'one' (line 165)
        one_6531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'one', False)
        # Applying the binary operator '-' (line 165)
        result_sub_6532 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 19), '-', temp1_6530, one_6531)
        
        # Getting the type of 'zero' (line 165)
        zero_6533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 34), 'zero', False)
        # Applying the binary operator '!=' (line 165)
        result_ne_6534 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 19), '!=', result_sub_6532, zero_6533)
        
        # Processing the call keyword arguments (line 165)
        kwargs_6535 = {}
        # Getting the type of 'any' (line 165)
        any_6529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'any', False)
        # Calling any(args, kwargs) (line 165)
        any_call_result_6536 = invoke(stypy.reporting.localization.Localization(__file__, 165, 15), any_6529, *[result_ne_6534], **kwargs_6535)
        
        # Testing the type of an if condition (line 165)
        if_condition_6537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 12), any_call_result_6536)
        # Assigning a type to the variable 'if_condition_6537' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'if_condition_6537', if_condition_6537)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 161)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to RuntimeError(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'msg' (line 168)
        msg_6539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 168)
        tuple_6540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 168)
        # Adding element type (line 168)
        # Getting the type of '_' (line 168)
        __6541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), '_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 38), tuple_6540, __6541)
        # Adding element type (line 168)
        # Getting the type of 'one' (line 168)
        one_6542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 41), 'one', False)
        # Obtaining the member 'dtype' of a type (line 168)
        dtype_6543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 41), one_6542, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 38), tuple_6540, dtype_6543)
        
        # Applying the binary operator '%' (line 168)
        result_mod_6544 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 31), '%', msg_6539, tuple_6540)
        
        # Processing the call keyword arguments (line 168)
        kwargs_6545 = {}
        # Getting the type of 'RuntimeError' (line 168)
        RuntimeError_6538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 168)
        RuntimeError_call_result_6546 = invoke(stypy.reporting.localization.Localization(__file__, 168, 18), RuntimeError_6538, *[result_mod_6544], **kwargs_6545)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 168, 12), RuntimeError_call_result_6546, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 169):
        # Getting the type of 'a' (line 169)
        a_6547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'a')
        # Getting the type of 'betah' (line 169)
        betah_6548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'betah')
        # Applying the binary operator '+' (line 169)
        result_add_6549 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), '+', a_6547, betah_6548)
        
        # Assigning a type to the variable 'temp' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'temp', result_add_6549)
        
        # Assigning a Num to a Name (line 170):
        int_6550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 15), 'int')
        # Assigning a type to the variable 'irnd' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'irnd', int_6550)
        
        
        # Call to any(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Getting the type of 'temp' (line 171)
        temp_6552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'temp', False)
        # Getting the type of 'a' (line 171)
        a_6553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'a', False)
        # Applying the binary operator '-' (line 171)
        result_sub_6554 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 15), '-', temp_6552, a_6553)
        
        # Getting the type of 'zero' (line 171)
        zero_6555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'zero', False)
        # Applying the binary operator '!=' (line 171)
        result_ne_6556 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 15), '!=', result_sub_6554, zero_6555)
        
        # Processing the call keyword arguments (line 171)
        kwargs_6557 = {}
        # Getting the type of 'any' (line 171)
        any_6551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'any', False)
        # Calling any(args, kwargs) (line 171)
        any_call_result_6558 = invoke(stypy.reporting.localization.Localization(__file__, 171, 11), any_6551, *[result_ne_6556], **kwargs_6557)
        
        # Testing the type of an if condition (line 171)
        if_condition_6559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 8), any_call_result_6558)
        # Assigning a type to the variable 'if_condition_6559' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'if_condition_6559', if_condition_6559)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 172):
        int_6560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 19), 'int')
        # Assigning a type to the variable 'irnd' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'irnd', int_6560)
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 173):
        # Getting the type of 'a' (line 173)
        a_6561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'a')
        # Getting the type of 'beta' (line 173)
        beta_6562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'beta')
        # Applying the binary operator '+' (line 173)
        result_add_6563 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '+', a_6561, beta_6562)
        
        # Assigning a type to the variable 'tempa' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'tempa', result_add_6563)
        
        # Assigning a BinOp to a Name (line 174):
        # Getting the type of 'tempa' (line 174)
        tempa_6564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'tempa')
        # Getting the type of 'betah' (line 174)
        betah_6565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'betah')
        # Applying the binary operator '+' (line 174)
        result_add_6566 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '+', tempa_6564, betah_6565)
        
        # Assigning a type to the variable 'temp' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'temp', result_add_6566)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'irnd' (line 175)
        irnd_6567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'irnd')
        int_6568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 19), 'int')
        # Applying the binary operator '==' (line 175)
        result_eq_6569 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), '==', irnd_6567, int_6568)
        
        
        # Call to any(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Getting the type of 'temp' (line 175)
        temp_6571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 29), 'temp', False)
        # Getting the type of 'tempa' (line 175)
        tempa_6572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 34), 'tempa', False)
        # Applying the binary operator '-' (line 175)
        result_sub_6573 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 29), '-', temp_6571, tempa_6572)
        
        # Getting the type of 'zero' (line 175)
        zero_6574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'zero', False)
        # Applying the binary operator '!=' (line 175)
        result_ne_6575 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 29), '!=', result_sub_6573, zero_6574)
        
        # Processing the call keyword arguments (line 175)
        kwargs_6576 = {}
        # Getting the type of 'any' (line 175)
        any_6570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'any', False)
        # Calling any(args, kwargs) (line 175)
        any_call_result_6577 = invoke(stypy.reporting.localization.Localization(__file__, 175, 25), any_6570, *[result_ne_6575], **kwargs_6576)
        
        # Applying the binary operator 'and' (line 175)
        result_and_keyword_6578 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), 'and', result_eq_6569, any_call_result_6577)
        
        # Testing the type of an if condition (line 175)
        if_condition_6579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), result_and_keyword_6578)
        # Assigning a type to the variable 'if_condition_6579' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_6579', if_condition_6579)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 176):
        int_6580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'int')
        # Assigning a type to the variable 'irnd' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'irnd', int_6580)
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 179):
        # Getting the type of 'it' (line 179)
        it_6581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'it')
        int_6582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'int')
        # Applying the binary operator '+' (line 179)
        result_add_6583 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 16), '+', it_6581, int_6582)
        
        # Assigning a type to the variable 'negep' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'negep', result_add_6583)
        
        # Assigning a BinOp to a Name (line 180):
        # Getting the type of 'one' (line 180)
        one_6584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'one')
        # Getting the type of 'beta' (line 180)
        beta_6585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'beta')
        # Applying the binary operator 'div' (line 180)
        result_div_6586 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 17), 'div', one_6584, beta_6585)
        
        # Assigning a type to the variable 'betain' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'betain', result_div_6586)
        
        # Assigning a Name to a Name (line 181):
        # Getting the type of 'one' (line 181)
        one_6587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'one')
        # Assigning a type to the variable 'a' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'a', one_6587)
        
        
        # Call to range(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'negep' (line 182)
        negep_6589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 23), 'negep', False)
        # Processing the call keyword arguments (line 182)
        kwargs_6590 = {}
        # Getting the type of 'range' (line 182)
        range_6588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'range', False)
        # Calling range(args, kwargs) (line 182)
        range_call_result_6591 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), range_6588, *[negep_6589], **kwargs_6590)
        
        # Testing the type of a for loop iterable (line 182)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 8), range_call_result_6591)
        # Getting the type of the for loop variable (line 182)
        for_loop_var_6592 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 8), range_call_result_6591)
        # Assigning a type to the variable 'i' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'i', for_loop_var_6592)
        # SSA begins for a for statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 183):
        # Getting the type of 'a' (line 183)
        a_6593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'a')
        # Getting the type of 'betain' (line 183)
        betain_6594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'betain')
        # Applying the binary operator '*' (line 183)
        result_mul_6595 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 16), '*', a_6593, betain_6594)
        
        # Assigning a type to the variable 'a' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'a', result_mul_6595)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'a' (line 184)
        a_6596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'a')
        # Assigning a type to the variable 'b' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'b', a_6596)
        
        
        # Call to range(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'max_iterN' (line 185)
        max_iterN_6598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'max_iterN', False)
        # Processing the call keyword arguments (line 185)
        kwargs_6599 = {}
        # Getting the type of 'range' (line 185)
        range_6597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'range', False)
        # Calling range(args, kwargs) (line 185)
        range_call_result_6600 = invoke(stypy.reporting.localization.Localization(__file__, 185, 17), range_6597, *[max_iterN_6598], **kwargs_6599)
        
        # Testing the type of a for loop iterable (line 185)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 8), range_call_result_6600)
        # Getting the type of the for loop variable (line 185)
        for_loop_var_6601 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 8), range_call_result_6600)
        # Assigning a type to the variable '_' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), '_', for_loop_var_6601)
        # SSA begins for a for statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 186):
        # Getting the type of 'one' (line 186)
        one_6602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'one')
        # Getting the type of 'a' (line 186)
        a_6603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'a')
        # Applying the binary operator '-' (line 186)
        result_sub_6604 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 19), '-', one_6602, a_6603)
        
        # Assigning a type to the variable 'temp' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'temp', result_sub_6604)
        
        
        # Call to any(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Getting the type of 'temp' (line 187)
        temp_6606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'temp', False)
        # Getting the type of 'one' (line 187)
        one_6607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'one', False)
        # Applying the binary operator '-' (line 187)
        result_sub_6608 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 19), '-', temp_6606, one_6607)
        
        # Getting the type of 'zero' (line 187)
        zero_6609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'zero', False)
        # Applying the binary operator '!=' (line 187)
        result_ne_6610 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 19), '!=', result_sub_6608, zero_6609)
        
        # Processing the call keyword arguments (line 187)
        kwargs_6611 = {}
        # Getting the type of 'any' (line 187)
        any_6605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'any', False)
        # Calling any(args, kwargs) (line 187)
        any_call_result_6612 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), any_6605, *[result_ne_6610], **kwargs_6611)
        
        # Testing the type of an if condition (line 187)
        if_condition_6613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 12), any_call_result_6612)
        # Assigning a type to the variable 'if_condition_6613' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'if_condition_6613', if_condition_6613)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 189):
        # Getting the type of 'a' (line 189)
        a_6614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'a')
        # Getting the type of 'beta' (line 189)
        beta_6615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'beta')
        # Applying the binary operator '*' (line 189)
        result_mul_6616 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 16), '*', a_6614, beta_6615)
        
        # Assigning a type to the variable 'a' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'a', result_mul_6616)
        
        # Assigning a BinOp to a Name (line 190):
        # Getting the type of 'negep' (line 190)
        negep_6617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'negep')
        int_6618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 28), 'int')
        # Applying the binary operator '-' (line 190)
        result_sub_6619 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 20), '-', negep_6617, int_6618)
        
        # Assigning a type to the variable 'negep' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'negep', result_sub_6619)
        
        
        # Getting the type of 'negep' (line 192)
        negep_6620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'negep')
        int_6621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'int')
        # Applying the binary operator '<' (line 192)
        result_lt_6622 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 15), '<', negep_6620, int_6621)
        
        # Testing the type of an if condition (line 192)
        if_condition_6623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 12), result_lt_6622)
        # Assigning a type to the variable 'if_condition_6623' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'if_condition_6623', if_condition_6623)
        # SSA begins for if statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 193)
        # Processing the call arguments (line 193)
        str_6625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 35), 'str', "could not determine machine tolerance for 'negep', locals() -> %s")
        
        # Call to locals(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_6627 = {}
        # Getting the type of 'locals' (line 194)
        locals_6626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 68), 'locals', False)
        # Calling locals(args, kwargs) (line 194)
        locals_call_result_6628 = invoke(stypy.reporting.localization.Localization(__file__, 194, 68), locals_6626, *[], **kwargs_6627)
        
        # Applying the binary operator '%' (line 193)
        result_mod_6629 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 35), '%', str_6625, locals_call_result_6628)
        
        # Processing the call keyword arguments (line 193)
        kwargs_6630 = {}
        # Getting the type of 'RuntimeError' (line 193)
        RuntimeError_6624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 193)
        RuntimeError_call_result_6631 = invoke(stypy.reporting.localization.Localization(__file__, 193, 22), RuntimeError_6624, *[result_mod_6629], **kwargs_6630)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 193, 16), RuntimeError_call_result_6631, 'raise parameter', BaseException)
        # SSA join for if statement (line 192)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 185)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to RuntimeError(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'msg' (line 196)
        msg_6633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_6634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        # Getting the type of '_' (line 196)
        __6635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 38), '_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 38), tuple_6634, __6635)
        # Adding element type (line 196)
        # Getting the type of 'one' (line 196)
        one_6636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 41), 'one', False)
        # Obtaining the member 'dtype' of a type (line 196)
        dtype_6637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 41), one_6636, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 38), tuple_6634, dtype_6637)
        
        # Applying the binary operator '%' (line 196)
        result_mod_6638 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 31), '%', msg_6633, tuple_6634)
        
        # Processing the call keyword arguments (line 196)
        kwargs_6639 = {}
        # Getting the type of 'RuntimeError' (line 196)
        RuntimeError_6632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 196)
        RuntimeError_call_result_6640 = invoke(stypy.reporting.localization.Localization(__file__, 196, 18), RuntimeError_6632, *[result_mod_6638], **kwargs_6639)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 196, 12), RuntimeError_call_result_6640, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a UnaryOp to a Name (line 197):
        
        # Getting the type of 'negep' (line 197)
        negep_6641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'negep')
        # Applying the 'usub' unary operator (line 197)
        result___neg___6642 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 16), 'usub', negep_6641)
        
        # Assigning a type to the variable 'negep' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'negep', result___neg___6642)
        
        # Assigning a Name to a Name (line 198):
        # Getting the type of 'a' (line 198)
        a_6643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'a')
        # Assigning a type to the variable 'epsneg' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'epsneg', a_6643)
        
        # Assigning a BinOp to a Name (line 201):
        
        # Getting the type of 'it' (line 201)
        it_6644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'it')
        # Applying the 'usub' unary operator (line 201)
        result___neg___6645 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 17), 'usub', it_6644)
        
        int_6646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 24), 'int')
        # Applying the binary operator '-' (line 201)
        result_sub_6647 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 17), '-', result___neg___6645, int_6646)
        
        # Assigning a type to the variable 'machep' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'machep', result_sub_6647)
        
        # Assigning a Name to a Name (line 202):
        # Getting the type of 'b' (line 202)
        b_6648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'b')
        # Assigning a type to the variable 'a' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'a', b_6648)
        
        
        # Call to range(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'max_iterN' (line 204)
        max_iterN_6650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'max_iterN', False)
        # Processing the call keyword arguments (line 204)
        kwargs_6651 = {}
        # Getting the type of 'range' (line 204)
        range_6649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 17), 'range', False)
        # Calling range(args, kwargs) (line 204)
        range_call_result_6652 = invoke(stypy.reporting.localization.Localization(__file__, 204, 17), range_6649, *[max_iterN_6650], **kwargs_6651)
        
        # Testing the type of a for loop iterable (line 204)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 204, 8), range_call_result_6652)
        # Getting the type of the for loop variable (line 204)
        for_loop_var_6653 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 204, 8), range_call_result_6652)
        # Assigning a type to the variable '_' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), '_', for_loop_var_6653)
        # SSA begins for a for statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 205):
        # Getting the type of 'one' (line 205)
        one_6654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'one')
        # Getting the type of 'a' (line 205)
        a_6655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 25), 'a')
        # Applying the binary operator '+' (line 205)
        result_add_6656 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 19), '+', one_6654, a_6655)
        
        # Assigning a type to the variable 'temp' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'temp', result_add_6656)
        
        
        # Call to any(...): (line 206)
        # Processing the call arguments (line 206)
        
        # Getting the type of 'temp' (line 206)
        temp_6658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'temp', False)
        # Getting the type of 'one' (line 206)
        one_6659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'one', False)
        # Applying the binary operator '-' (line 206)
        result_sub_6660 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 19), '-', temp_6658, one_6659)
        
        # Getting the type of 'zero' (line 206)
        zero_6661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 31), 'zero', False)
        # Applying the binary operator '!=' (line 206)
        result_ne_6662 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 19), '!=', result_sub_6660, zero_6661)
        
        # Processing the call keyword arguments (line 206)
        kwargs_6663 = {}
        # Getting the type of 'any' (line 206)
        any_6657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'any', False)
        # Calling any(args, kwargs) (line 206)
        any_call_result_6664 = invoke(stypy.reporting.localization.Localization(__file__, 206, 15), any_6657, *[result_ne_6662], **kwargs_6663)
        
        # Testing the type of an if condition (line 206)
        if_condition_6665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 12), any_call_result_6664)
        # Assigning a type to the variable 'if_condition_6665' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'if_condition_6665', if_condition_6665)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 208):
        # Getting the type of 'a' (line 208)
        a_6666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'a')
        # Getting the type of 'beta' (line 208)
        beta_6667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'beta')
        # Applying the binary operator '*' (line 208)
        result_mul_6668 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 16), '*', a_6666, beta_6667)
        
        # Assigning a type to the variable 'a' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'a', result_mul_6668)
        
        # Assigning a BinOp to a Name (line 209):
        # Getting the type of 'machep' (line 209)
        machep_6669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'machep')
        int_6670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 30), 'int')
        # Applying the binary operator '+' (line 209)
        result_add_6671 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 21), '+', machep_6669, int_6670)
        
        # Assigning a type to the variable 'machep' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'machep', result_add_6671)
        # SSA branch for the else part of a for statement (line 204)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to RuntimeError(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'msg' (line 211)
        msg_6673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 31), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 211)
        tuple_6674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 211)
        # Adding element type (line 211)
        # Getting the type of '_' (line 211)
        __6675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), '_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 38), tuple_6674, __6675)
        # Adding element type (line 211)
        # Getting the type of 'one' (line 211)
        one_6676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'one', False)
        # Obtaining the member 'dtype' of a type (line 211)
        dtype_6677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 41), one_6676, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 38), tuple_6674, dtype_6677)
        
        # Applying the binary operator '%' (line 211)
        result_mod_6678 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 31), '%', msg_6673, tuple_6674)
        
        # Processing the call keyword arguments (line 211)
        kwargs_6679 = {}
        # Getting the type of 'RuntimeError' (line 211)
        RuntimeError_6672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 211)
        RuntimeError_call_result_6680 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), RuntimeError_6672, *[result_mod_6678], **kwargs_6679)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 211, 12), RuntimeError_call_result_6680, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'a' (line 212)
        a_6681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'a')
        # Assigning a type to the variable 'eps' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'eps', a_6681)
        
        # Assigning a Num to a Name (line 215):
        int_6682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 15), 'int')
        # Assigning a type to the variable 'ngrd' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'ngrd', int_6682)
        
        # Assigning a BinOp to a Name (line 216):
        # Getting the type of 'one' (line 216)
        one_6683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'one')
        # Getting the type of 'eps' (line 216)
        eps_6684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'eps')
        # Applying the binary operator '+' (line 216)
        result_add_6685 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 15), '+', one_6683, eps_6684)
        
        # Assigning a type to the variable 'temp' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'temp', result_add_6685)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'irnd' (line 217)
        irnd_6686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'irnd')
        int_6687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 19), 'int')
        # Applying the binary operator '==' (line 217)
        result_eq_6688 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 11), '==', irnd_6686, int_6687)
        
        
        # Call to any(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Getting the type of 'temp' (line 217)
        temp_6690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'temp', False)
        # Getting the type of 'one' (line 217)
        one_6691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'one', False)
        # Applying the binary operator '*' (line 217)
        result_mul_6692 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 29), '*', temp_6690, one_6691)
        
        # Getting the type of 'one' (line 217)
        one_6693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 40), 'one', False)
        # Applying the binary operator '-' (line 217)
        result_sub_6694 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 29), '-', result_mul_6692, one_6693)
        
        # Getting the type of 'zero' (line 217)
        zero_6695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 47), 'zero', False)
        # Applying the binary operator '!=' (line 217)
        result_ne_6696 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 29), '!=', result_sub_6694, zero_6695)
        
        # Processing the call keyword arguments (line 217)
        kwargs_6697 = {}
        # Getting the type of 'any' (line 217)
        any_6689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'any', False)
        # Calling any(args, kwargs) (line 217)
        any_call_result_6698 = invoke(stypy.reporting.localization.Localization(__file__, 217, 25), any_6689, *[result_ne_6696], **kwargs_6697)
        
        # Applying the binary operator 'and' (line 217)
        result_and_keyword_6699 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 11), 'and', result_eq_6688, any_call_result_6698)
        
        # Testing the type of an if condition (line 217)
        if_condition_6700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 8), result_and_keyword_6699)
        # Assigning a type to the variable 'if_condition_6700' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'if_condition_6700', if_condition_6700)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 218):
        int_6701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 19), 'int')
        # Assigning a type to the variable 'ngrd' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'ngrd', int_6701)
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 221):
        int_6702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 12), 'int')
        # Assigning a type to the variable 'i' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'i', int_6702)
        
        # Assigning a Num to a Name (line 222):
        int_6703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 12), 'int')
        # Assigning a type to the variable 'k' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'k', int_6703)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'betain' (line 223)
        betain_6704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'betain')
        # Assigning a type to the variable 'z' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'z', betain_6704)
        
        # Assigning a BinOp to a Name (line 224):
        # Getting the type of 'one' (line 224)
        one_6705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'one')
        # Getting the type of 'eps' (line 224)
        eps_6706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 18), 'eps')
        # Applying the binary operator '+' (line 224)
        result_add_6707 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 12), '+', one_6705, eps_6706)
        
        # Assigning a type to the variable 't' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 't', result_add_6707)
        
        # Assigning a Num to a Name (line 225):
        int_6708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 16), 'int')
        # Assigning a type to the variable 'nxres' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'nxres', int_6708)
        
        
        # Call to range(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'max_iterN' (line 226)
        max_iterN_6710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'max_iterN', False)
        # Processing the call keyword arguments (line 226)
        kwargs_6711 = {}
        # Getting the type of 'range' (line 226)
        range_6709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'range', False)
        # Calling range(args, kwargs) (line 226)
        range_call_result_6712 = invoke(stypy.reporting.localization.Localization(__file__, 226, 17), range_6709, *[max_iterN_6710], **kwargs_6711)
        
        # Testing the type of a for loop iterable (line 226)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 8), range_call_result_6712)
        # Getting the type of the for loop variable (line 226)
        for_loop_var_6713 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 8), range_call_result_6712)
        # Assigning a type to the variable '_' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), '_', for_loop_var_6713)
        # SSA begins for a for statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 227):
        # Getting the type of 'z' (line 227)
        z_6714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'z')
        # Assigning a type to the variable 'y' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'y', z_6714)
        
        # Assigning a BinOp to a Name (line 228):
        # Getting the type of 'y' (line 228)
        y_6715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'y')
        # Getting the type of 'y' (line 228)
        y_6716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'y')
        # Applying the binary operator '*' (line 228)
        result_mul_6717 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 16), '*', y_6715, y_6716)
        
        # Assigning a type to the variable 'z' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'z', result_mul_6717)
        
        # Assigning a BinOp to a Name (line 229):
        # Getting the type of 'z' (line 229)
        z_6718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'z')
        # Getting the type of 'one' (line 229)
        one_6719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 18), 'one')
        # Applying the binary operator '*' (line 229)
        result_mul_6720 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 16), '*', z_6718, one_6719)
        
        # Assigning a type to the variable 'a' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'a', result_mul_6720)
        
        # Assigning a BinOp to a Name (line 230):
        # Getting the type of 'z' (line 230)
        z_6721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'z')
        # Getting the type of 't' (line 230)
        t_6722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 't')
        # Applying the binary operator '*' (line 230)
        result_mul_6723 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 19), '*', z_6721, t_6722)
        
        # Assigning a type to the variable 'temp' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'temp', result_mul_6723)
        
        
        # Evaluating a boolean operation
        
        # Call to any(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Getting the type of 'a' (line 231)
        a_6725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'a', False)
        # Getting the type of 'a' (line 231)
        a_6726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 'a', False)
        # Applying the binary operator '+' (line 231)
        result_add_6727 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 19), '+', a_6725, a_6726)
        
        # Getting the type of 'zero' (line 231)
        zero_6728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 26), 'zero', False)
        # Applying the binary operator '==' (line 231)
        result_eq_6729 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 19), '==', result_add_6727, zero_6728)
        
        # Processing the call keyword arguments (line 231)
        kwargs_6730 = {}
        # Getting the type of 'any' (line 231)
        any_6724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'any', False)
        # Calling any(args, kwargs) (line 231)
        any_call_result_6731 = invoke(stypy.reporting.localization.Localization(__file__, 231, 15), any_6724, *[result_eq_6729], **kwargs_6730)
        
        
        # Call to any(...): (line 231)
        # Processing the call arguments (line 231)
        
        
        # Call to abs(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'z' (line 231)
        z_6734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'z', False)
        # Processing the call keyword arguments (line 231)
        kwargs_6735 = {}
        # Getting the type of 'abs' (line 231)
        abs_6733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 39), 'abs', False)
        # Calling abs(args, kwargs) (line 231)
        abs_call_result_6736 = invoke(stypy.reporting.localization.Localization(__file__, 231, 39), abs_6733, *[z_6734], **kwargs_6735)
        
        # Getting the type of 'y' (line 231)
        y_6737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 49), 'y', False)
        # Applying the binary operator '>=' (line 231)
        result_ge_6738 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 39), '>=', abs_call_result_6736, y_6737)
        
        # Processing the call keyword arguments (line 231)
        kwargs_6739 = {}
        # Getting the type of 'any' (line 231)
        any_6732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 35), 'any', False)
        # Calling any(args, kwargs) (line 231)
        any_call_result_6740 = invoke(stypy.reporting.localization.Localization(__file__, 231, 35), any_6732, *[result_ge_6738], **kwargs_6739)
        
        # Applying the binary operator 'or' (line 231)
        result_or_keyword_6741 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 15), 'or', any_call_result_6731, any_call_result_6740)
        
        # Testing the type of an if condition (line 231)
        if_condition_6742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 12), result_or_keyword_6741)
        # Assigning a type to the variable 'if_condition_6742' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'if_condition_6742', if_condition_6742)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 233):
        # Getting the type of 'temp' (line 233)
        temp_6743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'temp')
        # Getting the type of 'betain' (line 233)
        betain_6744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'betain')
        # Applying the binary operator '*' (line 233)
        result_mul_6745 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 20), '*', temp_6743, betain_6744)
        
        # Assigning a type to the variable 'temp1' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'temp1', result_mul_6745)
        
        
        # Call to any(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Getting the type of 'temp1' (line 234)
        temp1_6747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'temp1', False)
        # Getting the type of 'beta' (line 234)
        beta_6748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'beta', False)
        # Applying the binary operator '*' (line 234)
        result_mul_6749 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 19), '*', temp1_6747, beta_6748)
        
        # Getting the type of 'z' (line 234)
        z_6750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 33), 'z', False)
        # Applying the binary operator '==' (line 234)
        result_eq_6751 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 19), '==', result_mul_6749, z_6750)
        
        # Processing the call keyword arguments (line 234)
        kwargs_6752 = {}
        # Getting the type of 'any' (line 234)
        any_6746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'any', False)
        # Calling any(args, kwargs) (line 234)
        any_call_result_6753 = invoke(stypy.reporting.localization.Localization(__file__, 234, 15), any_6746, *[result_eq_6751], **kwargs_6752)
        
        # Testing the type of an if condition (line 234)
        if_condition_6754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 12), any_call_result_6753)
        # Assigning a type to the variable 'if_condition_6754' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'if_condition_6754', if_condition_6754)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 236):
        # Getting the type of 'i' (line 236)
        i_6755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'i')
        int_6756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 20), 'int')
        # Applying the binary operator '+' (line 236)
        result_add_6757 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 16), '+', i_6755, int_6756)
        
        # Assigning a type to the variable 'i' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'i', result_add_6757)
        
        # Assigning a BinOp to a Name (line 237):
        # Getting the type of 'k' (line 237)
        k_6758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'k')
        # Getting the type of 'k' (line 237)
        k_6759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'k')
        # Applying the binary operator '+' (line 237)
        result_add_6760 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 16), '+', k_6758, k_6759)
        
        # Assigning a type to the variable 'k' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'k', result_add_6760)
        # SSA branch for the else part of a for statement (line 226)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to RuntimeError(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'msg' (line 239)
        msg_6762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 239)
        tuple_6763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 239)
        # Adding element type (line 239)
        # Getting the type of '_' (line 239)
        __6764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 38), '_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 38), tuple_6763, __6764)
        # Adding element type (line 239)
        # Getting the type of 'one' (line 239)
        one_6765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 41), 'one', False)
        # Obtaining the member 'dtype' of a type (line 239)
        dtype_6766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 41), one_6765, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 38), tuple_6763, dtype_6766)
        
        # Applying the binary operator '%' (line 239)
        result_mod_6767 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 31), '%', msg_6762, tuple_6763)
        
        # Processing the call keyword arguments (line 239)
        kwargs_6768 = {}
        # Getting the type of 'RuntimeError' (line 239)
        RuntimeError_6761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 239)
        RuntimeError_call_result_6769 = invoke(stypy.reporting.localization.Localization(__file__, 239, 18), RuntimeError_6761, *[result_mod_6767], **kwargs_6768)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 239, 12), RuntimeError_call_result_6769, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ibeta' (line 240)
        ibeta_6770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'ibeta')
        int_6771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 20), 'int')
        # Applying the binary operator '!=' (line 240)
        result_ne_6772 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), '!=', ibeta_6770, int_6771)
        
        # Testing the type of an if condition (line 240)
        if_condition_6773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_ne_6772)
        # Assigning a type to the variable 'if_condition_6773' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_6773', if_condition_6773)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 241):
        # Getting the type of 'i' (line 241)
        i_6774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 19), 'i')
        int_6775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 23), 'int')
        # Applying the binary operator '+' (line 241)
        result_add_6776 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 19), '+', i_6774, int_6775)
        
        # Assigning a type to the variable 'iexp' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'iexp', result_add_6776)
        
        # Assigning a BinOp to a Name (line 242):
        # Getting the type of 'k' (line 242)
        k_6777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 17), 'k')
        # Getting the type of 'k' (line 242)
        k_6778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'k')
        # Applying the binary operator '+' (line 242)
        result_add_6779 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 17), '+', k_6777, k_6778)
        
        # Assigning a type to the variable 'mx' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'mx', result_add_6779)
        # SSA branch for the else part of an if statement (line 240)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 244):
        int_6780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 19), 'int')
        # Assigning a type to the variable 'iexp' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'iexp', int_6780)
        
        # Assigning a Name to a Name (line 245):
        # Getting the type of 'ibeta' (line 245)
        ibeta_6781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'ibeta')
        # Assigning a type to the variable 'iz' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'iz', ibeta_6781)
        
        
        # Getting the type of 'k' (line 246)
        k_6782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 18), 'k')
        # Getting the type of 'iz' (line 246)
        iz_6783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'iz')
        # Applying the binary operator '>=' (line 246)
        result_ge_6784 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 18), '>=', k_6782, iz_6783)
        
        # Testing the type of an if condition (line 246)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 12), result_ge_6784)
        # SSA begins for while statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 247):
        # Getting the type of 'iz' (line 247)
        iz_6785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'iz')
        # Getting the type of 'ibeta' (line 247)
        ibeta_6786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 26), 'ibeta')
        # Applying the binary operator '*' (line 247)
        result_mul_6787 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 21), '*', iz_6785, ibeta_6786)
        
        # Assigning a type to the variable 'iz' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'iz', result_mul_6787)
        
        # Assigning a BinOp to a Name (line 248):
        # Getting the type of 'iexp' (line 248)
        iexp_6788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 23), 'iexp')
        int_6789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 30), 'int')
        # Applying the binary operator '+' (line 248)
        result_add_6790 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 23), '+', iexp_6788, int_6789)
        
        # Assigning a type to the variable 'iexp' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'iexp', result_add_6790)
        # SSA join for while statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 249):
        # Getting the type of 'iz' (line 249)
        iz_6791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 17), 'iz')
        # Getting the type of 'iz' (line 249)
        iz_6792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 22), 'iz')
        # Applying the binary operator '+' (line 249)
        result_add_6793 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 17), '+', iz_6791, iz_6792)
        
        int_6794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 27), 'int')
        # Applying the binary operator '-' (line 249)
        result_sub_6795 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 25), '-', result_add_6793, int_6794)
        
        # Assigning a type to the variable 'mx' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'mx', result_sub_6795)
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to range(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'max_iterN' (line 252)
        max_iterN_6797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 23), 'max_iterN', False)
        # Processing the call keyword arguments (line 252)
        kwargs_6798 = {}
        # Getting the type of 'range' (line 252)
        range_6796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 17), 'range', False)
        # Calling range(args, kwargs) (line 252)
        range_call_result_6799 = invoke(stypy.reporting.localization.Localization(__file__, 252, 17), range_6796, *[max_iterN_6797], **kwargs_6798)
        
        # Testing the type of a for loop iterable (line 252)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 252, 8), range_call_result_6799)
        # Getting the type of the for loop variable (line 252)
        for_loop_var_6800 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 252, 8), range_call_result_6799)
        # Assigning a type to the variable '_' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), '_', for_loop_var_6800)
        # SSA begins for a for statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'y' (line 253)
        y_6801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'y')
        # Assigning a type to the variable 'xmin' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'xmin', y_6801)
        
        # Assigning a BinOp to a Name (line 254):
        # Getting the type of 'y' (line 254)
        y_6802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'y')
        # Getting the type of 'betain' (line 254)
        betain_6803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'betain')
        # Applying the binary operator '*' (line 254)
        result_mul_6804 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 16), '*', y_6802, betain_6803)
        
        # Assigning a type to the variable 'y' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'y', result_mul_6804)
        
        # Assigning a BinOp to a Name (line 255):
        # Getting the type of 'y' (line 255)
        y_6805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'y')
        # Getting the type of 'one' (line 255)
        one_6806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'one')
        # Applying the binary operator '*' (line 255)
        result_mul_6807 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 16), '*', y_6805, one_6806)
        
        # Assigning a type to the variable 'a' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'a', result_mul_6807)
        
        # Assigning a BinOp to a Name (line 256):
        # Getting the type of 'y' (line 256)
        y_6808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'y')
        # Getting the type of 't' (line 256)
        t_6809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 't')
        # Applying the binary operator '*' (line 256)
        result_mul_6810 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 19), '*', y_6808, t_6809)
        
        # Assigning a type to the variable 'temp' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'temp', result_mul_6810)
        
        
        # Evaluating a boolean operation
        
        # Call to any(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Getting the type of 'a' (line 257)
        a_6812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'a', False)
        # Getting the type of 'a' (line 257)
        a_6813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 24), 'a', False)
        # Applying the binary operator '+' (line 257)
        result_add_6814 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 20), '+', a_6812, a_6813)
        
        # Getting the type of 'zero' (line 257)
        zero_6815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 30), 'zero', False)
        # Applying the binary operator '!=' (line 257)
        result_ne_6816 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 19), '!=', result_add_6814, zero_6815)
        
        # Processing the call keyword arguments (line 257)
        kwargs_6817 = {}
        # Getting the type of 'any' (line 257)
        any_6811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'any', False)
        # Calling any(args, kwargs) (line 257)
        any_call_result_6818 = invoke(stypy.reporting.localization.Localization(__file__, 257, 15), any_6811, *[result_ne_6816], **kwargs_6817)
        
        
        # Call to any(...): (line 257)
        # Processing the call arguments (line 257)
        
        
        # Call to abs(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'y' (line 257)
        y_6821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 48), 'y', False)
        # Processing the call keyword arguments (line 257)
        kwargs_6822 = {}
        # Getting the type of 'abs' (line 257)
        abs_6820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 44), 'abs', False)
        # Calling abs(args, kwargs) (line 257)
        abs_call_result_6823 = invoke(stypy.reporting.localization.Localization(__file__, 257, 44), abs_6820, *[y_6821], **kwargs_6822)
        
        # Getting the type of 'xmin' (line 257)
        xmin_6824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 53), 'xmin', False)
        # Applying the binary operator '<' (line 257)
        result_lt_6825 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 44), '<', abs_call_result_6823, xmin_6824)
        
        # Processing the call keyword arguments (line 257)
        kwargs_6826 = {}
        # Getting the type of 'any' (line 257)
        any_6819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 40), 'any', False)
        # Calling any(args, kwargs) (line 257)
        any_call_result_6827 = invoke(stypy.reporting.localization.Localization(__file__, 257, 40), any_6819, *[result_lt_6825], **kwargs_6826)
        
        # Applying the binary operator 'and' (line 257)
        result_and_keyword_6828 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 15), 'and', any_call_result_6818, any_call_result_6827)
        
        # Testing the type of an if condition (line 257)
        if_condition_6829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 12), result_and_keyword_6828)
        # Assigning a type to the variable 'if_condition_6829' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'if_condition_6829', if_condition_6829)
        # SSA begins for if statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 258):
        # Getting the type of 'k' (line 258)
        k_6830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'k')
        int_6831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 24), 'int')
        # Applying the binary operator '+' (line 258)
        result_add_6832 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 20), '+', k_6830, int_6831)
        
        # Assigning a type to the variable 'k' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'k', result_add_6832)
        
        # Assigning a BinOp to a Name (line 259):
        # Getting the type of 'temp' (line 259)
        temp_6833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'temp')
        # Getting the type of 'betain' (line 259)
        betain_6834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'betain')
        # Applying the binary operator '*' (line 259)
        result_mul_6835 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 24), '*', temp_6833, betain_6834)
        
        # Assigning a type to the variable 'temp1' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'temp1', result_mul_6835)
        
        
        # Evaluating a boolean operation
        
        # Call to any(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Getting the type of 'temp1' (line 260)
        temp1_6837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'temp1', False)
        # Getting the type of 'beta' (line 260)
        beta_6838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 'beta', False)
        # Applying the binary operator '*' (line 260)
        result_mul_6839 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 23), '*', temp1_6837, beta_6838)
        
        # Getting the type of 'y' (line 260)
        y_6840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'y', False)
        # Applying the binary operator '==' (line 260)
        result_eq_6841 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 23), '==', result_mul_6839, y_6840)
        
        # Processing the call keyword arguments (line 260)
        kwargs_6842 = {}
        # Getting the type of 'any' (line 260)
        any_6836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 19), 'any', False)
        # Calling any(args, kwargs) (line 260)
        any_call_result_6843 = invoke(stypy.reporting.localization.Localization(__file__, 260, 19), any_6836, *[result_eq_6841], **kwargs_6842)
        
        
        # Call to any(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Getting the type of 'temp' (line 260)
        temp_6845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 48), 'temp', False)
        # Getting the type of 'y' (line 260)
        y_6846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 56), 'y', False)
        # Applying the binary operator '!=' (line 260)
        result_ne_6847 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 48), '!=', temp_6845, y_6846)
        
        # Processing the call keyword arguments (line 260)
        kwargs_6848 = {}
        # Getting the type of 'any' (line 260)
        any_6844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 44), 'any', False)
        # Calling any(args, kwargs) (line 260)
        any_call_result_6849 = invoke(stypy.reporting.localization.Localization(__file__, 260, 44), any_6844, *[result_ne_6847], **kwargs_6848)
        
        # Applying the binary operator 'and' (line 260)
        result_and_keyword_6850 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 19), 'and', any_call_result_6843, any_call_result_6849)
        
        # Testing the type of an if condition (line 260)
        if_condition_6851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 16), result_and_keyword_6850)
        # Assigning a type to the variable 'if_condition_6851' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'if_condition_6851', if_condition_6851)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 261):
        int_6852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 28), 'int')
        # Assigning a type to the variable 'nxres' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'nxres', int_6852)
        
        # Assigning a Name to a Name (line 262):
        # Getting the type of 'y' (line 262)
        y_6853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 27), 'y')
        # Assigning a type to the variable 'xmin' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'xmin', y_6853)
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 257)
        module_type_store.open_ssa_branch('else')
        # SSA join for if statement (line 257)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 252)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to RuntimeError(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'msg' (line 267)
        msg_6855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 31), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 267)
        tuple_6856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 267)
        # Adding element type (line 267)
        # Getting the type of '_' (line 267)
        __6857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 38), '_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 38), tuple_6856, __6857)
        # Adding element type (line 267)
        # Getting the type of 'one' (line 267)
        one_6858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 41), 'one', False)
        # Obtaining the member 'dtype' of a type (line 267)
        dtype_6859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 41), one_6858, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 38), tuple_6856, dtype_6859)
        
        # Applying the binary operator '%' (line 267)
        result_mod_6860 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 31), '%', msg_6855, tuple_6856)
        
        # Processing the call keyword arguments (line 267)
        kwargs_6861 = {}
        # Getting the type of 'RuntimeError' (line 267)
        RuntimeError_6854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 267)
        RuntimeError_call_result_6862 = invoke(stypy.reporting.localization.Localization(__file__, 267, 18), RuntimeError_6854, *[result_mod_6860], **kwargs_6861)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 267, 12), RuntimeError_call_result_6862, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a UnaryOp to a Name (line 268):
        
        # Getting the type of 'k' (line 268)
        k_6863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 18), 'k')
        # Applying the 'usub' unary operator (line 268)
        result___neg___6864 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 17), 'usub', k_6863)
        
        # Assigning a type to the variable 'minexp' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'minexp', result___neg___6864)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mx' (line 271)
        mx_6865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'mx')
        # Getting the type of 'k' (line 271)
        k_6866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 17), 'k')
        # Getting the type of 'k' (line 271)
        k_6867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'k')
        # Applying the binary operator '+' (line 271)
        result_add_6868 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 17), '+', k_6866, k_6867)
        
        int_6869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 25), 'int')
        # Applying the binary operator '-' (line 271)
        result_sub_6870 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 23), '-', result_add_6868, int_6869)
        
        # Applying the binary operator '<=' (line 271)
        result_le_6871 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 11), '<=', mx_6865, result_sub_6870)
        
        
        # Getting the type of 'ibeta' (line 271)
        ibeta_6872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 31), 'ibeta')
        int_6873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 40), 'int')
        # Applying the binary operator '!=' (line 271)
        result_ne_6874 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 31), '!=', ibeta_6872, int_6873)
        
        # Applying the binary operator 'and' (line 271)
        result_and_keyword_6875 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 11), 'and', result_le_6871, result_ne_6874)
        
        # Testing the type of an if condition (line 271)
        if_condition_6876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 8), result_and_keyword_6875)
        # Assigning a type to the variable 'if_condition_6876' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'if_condition_6876', if_condition_6876)
        # SSA begins for if statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 272):
        # Getting the type of 'mx' (line 272)
        mx_6877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'mx')
        # Getting the type of 'mx' (line 272)
        mx_6878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'mx')
        # Applying the binary operator '+' (line 272)
        result_add_6879 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 17), '+', mx_6877, mx_6878)
        
        # Assigning a type to the variable 'mx' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'mx', result_add_6879)
        
        # Assigning a BinOp to a Name (line 273):
        # Getting the type of 'iexp' (line 273)
        iexp_6880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 19), 'iexp')
        int_6881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 26), 'int')
        # Applying the binary operator '+' (line 273)
        result_add_6882 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 19), '+', iexp_6880, int_6881)
        
        # Assigning a type to the variable 'iexp' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'iexp', result_add_6882)
        # SSA join for if statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 274):
        # Getting the type of 'mx' (line 274)
        mx_6883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 17), 'mx')
        # Getting the type of 'minexp' (line 274)
        minexp_6884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'minexp')
        # Applying the binary operator '+' (line 274)
        result_add_6885 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 17), '+', mx_6883, minexp_6884)
        
        # Assigning a type to the variable 'maxexp' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'maxexp', result_add_6885)
        
        # Assigning a BinOp to a Name (line 275):
        # Getting the type of 'irnd' (line 275)
        irnd_6886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'irnd')
        # Getting the type of 'nxres' (line 275)
        nxres_6887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), 'nxres')
        # Applying the binary operator '+' (line 275)
        result_add_6888 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 15), '+', irnd_6886, nxres_6887)
        
        # Assigning a type to the variable 'irnd' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'irnd', result_add_6888)
        
        
        # Getting the type of 'irnd' (line 276)
        irnd_6889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), 'irnd')
        int_6890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 19), 'int')
        # Applying the binary operator '>=' (line 276)
        result_ge_6891 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 11), '>=', irnd_6889, int_6890)
        
        # Testing the type of an if condition (line 276)
        if_condition_6892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 8), result_ge_6891)
        # Assigning a type to the variable 'if_condition_6892' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'if_condition_6892', if_condition_6892)
        # SSA begins for if statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 277):
        # Getting the type of 'maxexp' (line 277)
        maxexp_6893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'maxexp')
        int_6894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 30), 'int')
        # Applying the binary operator '-' (line 277)
        result_sub_6895 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 21), '-', maxexp_6893, int_6894)
        
        # Assigning a type to the variable 'maxexp' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'maxexp', result_sub_6895)
        # SSA join for if statement (line 276)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 278):
        # Getting the type of 'maxexp' (line 278)
        maxexp_6896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'maxexp')
        # Getting the type of 'minexp' (line 278)
        minexp_6897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'minexp')
        # Applying the binary operator '+' (line 278)
        result_add_6898 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 12), '+', maxexp_6896, minexp_6897)
        
        # Assigning a type to the variable 'i' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'i', result_add_6898)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ibeta' (line 279)
        ibeta_6899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'ibeta')
        int_6900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 20), 'int')
        # Applying the binary operator '==' (line 279)
        result_eq_6901 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), '==', ibeta_6899, int_6900)
        
        
        # Getting the type of 'i' (line 279)
        i_6902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 30), 'i')
        # Applying the 'not' unary operator (line 279)
        result_not__6903 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 26), 'not', i_6902)
        
        # Applying the binary operator 'and' (line 279)
        result_and_keyword_6904 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), 'and', result_eq_6901, result_not__6903)
        
        # Testing the type of an if condition (line 279)
        if_condition_6905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_and_keyword_6904)
        # Assigning a type to the variable 'if_condition_6905' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_6905', if_condition_6905)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 280):
        # Getting the type of 'maxexp' (line 280)
        maxexp_6906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 21), 'maxexp')
        int_6907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 30), 'int')
        # Applying the binary operator '-' (line 280)
        result_sub_6908 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 21), '-', maxexp_6906, int_6907)
        
        # Assigning a type to the variable 'maxexp' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'maxexp', result_sub_6908)
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'i' (line 281)
        i_6909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'i')
        int_6910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 15), 'int')
        # Applying the binary operator '>' (line 281)
        result_gt_6911 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 11), '>', i_6909, int_6910)
        
        # Testing the type of an if condition (line 281)
        if_condition_6912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 8), result_gt_6911)
        # Assigning a type to the variable 'if_condition_6912' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'if_condition_6912', if_condition_6912)
        # SSA begins for if statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 282):
        # Getting the type of 'maxexp' (line 282)
        maxexp_6913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 21), 'maxexp')
        int_6914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 30), 'int')
        # Applying the binary operator '-' (line 282)
        result_sub_6915 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 21), '-', maxexp_6913, int_6914)
        
        # Assigning a type to the variable 'maxexp' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'maxexp', result_sub_6915)
        # SSA join for if statement (line 281)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to any(...): (line 283)
        # Processing the call arguments (line 283)
        
        # Getting the type of 'a' (line 283)
        a_6917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'a', False)
        # Getting the type of 'y' (line 283)
        y_6918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'y', False)
        # Applying the binary operator '!=' (line 283)
        result_ne_6919 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 15), '!=', a_6917, y_6918)
        
        # Processing the call keyword arguments (line 283)
        kwargs_6920 = {}
        # Getting the type of 'any' (line 283)
        any_6916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'any', False)
        # Calling any(args, kwargs) (line 283)
        any_call_result_6921 = invoke(stypy.reporting.localization.Localization(__file__, 283, 11), any_6916, *[result_ne_6919], **kwargs_6920)
        
        # Testing the type of an if condition (line 283)
        if_condition_6922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 8), any_call_result_6921)
        # Assigning a type to the variable 'if_condition_6922' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'if_condition_6922', if_condition_6922)
        # SSA begins for if statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 284):
        # Getting the type of 'maxexp' (line 284)
        maxexp_6923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'maxexp')
        int_6924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 30), 'int')
        # Applying the binary operator '-' (line 284)
        result_sub_6925 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 21), '-', maxexp_6923, int_6924)
        
        # Assigning a type to the variable 'maxexp' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'maxexp', result_sub_6925)
        # SSA join for if statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 285):
        # Getting the type of 'one' (line 285)
        one_6926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'one')
        # Getting the type of 'epsneg' (line 285)
        epsneg_6927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'epsneg')
        # Applying the binary operator '-' (line 285)
        result_sub_6928 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 15), '-', one_6926, epsneg_6927)
        
        # Assigning a type to the variable 'xmax' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'xmax', result_sub_6928)
        
        
        # Call to any(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Getting the type of 'xmax' (line 286)
        xmax_6930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'xmax', False)
        # Getting the type of 'one' (line 286)
        one_6931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'one', False)
        # Applying the binary operator '*' (line 286)
        result_mul_6932 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 15), '*', xmax_6930, one_6931)
        
        # Getting the type of 'xmax' (line 286)
        xmax_6933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'xmax', False)
        # Applying the binary operator '!=' (line 286)
        result_ne_6934 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 15), '!=', result_mul_6932, xmax_6933)
        
        # Processing the call keyword arguments (line 286)
        kwargs_6935 = {}
        # Getting the type of 'any' (line 286)
        any_6929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'any', False)
        # Calling any(args, kwargs) (line 286)
        any_call_result_6936 = invoke(stypy.reporting.localization.Localization(__file__, 286, 11), any_6929, *[result_ne_6934], **kwargs_6935)
        
        # Testing the type of an if condition (line 286)
        if_condition_6937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 8), any_call_result_6936)
        # Assigning a type to the variable 'if_condition_6937' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'if_condition_6937', if_condition_6937)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 287):
        # Getting the type of 'one' (line 287)
        one_6938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'one')
        # Getting the type of 'beta' (line 287)
        beta_6939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'beta')
        # Getting the type of 'epsneg' (line 287)
        epsneg_6940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 30), 'epsneg')
        # Applying the binary operator '*' (line 287)
        result_mul_6941 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 25), '*', beta_6939, epsneg_6940)
        
        # Applying the binary operator '-' (line 287)
        result_sub_6942 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 19), '-', one_6938, result_mul_6941)
        
        # Assigning a type to the variable 'xmax' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'xmax', result_sub_6942)
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 288):
        # Getting the type of 'xmax' (line 288)
        xmax_6943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'xmax')
        # Getting the type of 'xmin' (line 288)
        xmin_6944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'xmin')
        # Getting the type of 'beta' (line 288)
        beta_6945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 28), 'beta')
        # Applying the binary operator '*' (line 288)
        result_mul_6946 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 23), '*', xmin_6944, beta_6945)
        
        # Getting the type of 'beta' (line 288)
        beta_6947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 33), 'beta')
        # Applying the binary operator '*' (line 288)
        result_mul_6948 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 32), '*', result_mul_6946, beta_6947)
        
        # Getting the type of 'beta' (line 288)
        beta_6949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 38), 'beta')
        # Applying the binary operator '*' (line 288)
        result_mul_6950 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 37), '*', result_mul_6948, beta_6949)
        
        # Applying the binary operator 'div' (line 288)
        result_div_6951 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 15), 'div', xmax_6943, result_mul_6950)
        
        # Assigning a type to the variable 'xmax' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'xmax', result_div_6951)
        
        # Assigning a BinOp to a Name (line 289):
        # Getting the type of 'maxexp' (line 289)
        maxexp_6952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'maxexp')
        # Getting the type of 'minexp' (line 289)
        minexp_6953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 21), 'minexp')
        # Applying the binary operator '+' (line 289)
        result_add_6954 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 12), '+', maxexp_6952, minexp_6953)
        
        int_6955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 30), 'int')
        # Applying the binary operator '+' (line 289)
        result_add_6956 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 28), '+', result_add_6954, int_6955)
        
        # Assigning a type to the variable 'i' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'i', result_add_6956)
        
        
        # Call to range(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'i' (line 290)
        i_6958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 23), 'i', False)
        # Processing the call keyword arguments (line 290)
        kwargs_6959 = {}
        # Getting the type of 'range' (line 290)
        range_6957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'range', False)
        # Calling range(args, kwargs) (line 290)
        range_call_result_6960 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), range_6957, *[i_6958], **kwargs_6959)
        
        # Testing the type of a for loop iterable (line 290)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 290, 8), range_call_result_6960)
        # Getting the type of the for loop variable (line 290)
        for_loop_var_6961 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 290, 8), range_call_result_6960)
        # Assigning a type to the variable 'j' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'j', for_loop_var_6961)
        # SSA begins for a for statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'ibeta' (line 291)
        ibeta_6962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'ibeta')
        int_6963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 24), 'int')
        # Applying the binary operator '==' (line 291)
        result_eq_6964 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 15), '==', ibeta_6962, int_6963)
        
        # Testing the type of an if condition (line 291)
        if_condition_6965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 12), result_eq_6964)
        # Assigning a type to the variable 'if_condition_6965' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'if_condition_6965', if_condition_6965)
        # SSA begins for if statement (line 291)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 292):
        # Getting the type of 'xmax' (line 292)
        xmax_6966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'xmax')
        # Getting the type of 'xmax' (line 292)
        xmax_6967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'xmax')
        # Applying the binary operator '+' (line 292)
        result_add_6968 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 23), '+', xmax_6966, xmax_6967)
        
        # Assigning a type to the variable 'xmax' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'xmax', result_add_6968)
        # SSA branch for the else part of an if statement (line 291)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 294):
        # Getting the type of 'xmax' (line 294)
        xmax_6969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'xmax')
        # Getting the type of 'beta' (line 294)
        beta_6970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 30), 'beta')
        # Applying the binary operator '*' (line 294)
        result_mul_6971 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 23), '*', xmax_6969, beta_6970)
        
        # Assigning a type to the variable 'xmax' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'xmax', result_mul_6971)
        # SSA join for if statement (line 291)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 296):
        # Getting the type of 'ibeta' (line 296)
        ibeta_6972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 'ibeta')
        # Getting the type of 'self' (line 296)
        self_6973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self')
        # Setting the type of the member 'ibeta' of a type (line 296)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_6973, 'ibeta', ibeta_6972)
        
        # Assigning a Name to a Attribute (line 297):
        # Getting the type of 'it' (line 297)
        it_6974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 18), 'it')
        # Getting the type of 'self' (line 297)
        self_6975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'self')
        # Setting the type of the member 'it' of a type (line 297)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), self_6975, 'it', it_6974)
        
        # Assigning a Name to a Attribute (line 298):
        # Getting the type of 'negep' (line 298)
        negep_6976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'negep')
        # Getting the type of 'self' (line 298)
        self_6977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'self')
        # Setting the type of the member 'negep' of a type (line 298)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), self_6977, 'negep', negep_6976)
        
        # Assigning a Call to a Attribute (line 299):
        
        # Call to float_to_float(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'epsneg' (line 299)
        epsneg_6979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 37), 'epsneg', False)
        # Processing the call keyword arguments (line 299)
        kwargs_6980 = {}
        # Getting the type of 'float_to_float' (line 299)
        float_to_float_6978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'float_to_float', False)
        # Calling float_to_float(args, kwargs) (line 299)
        float_to_float_call_result_6981 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), float_to_float_6978, *[epsneg_6979], **kwargs_6980)
        
        # Getting the type of 'self' (line 299)
        self_6982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'self')
        # Setting the type of the member 'epsneg' of a type (line 299)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), self_6982, 'epsneg', float_to_float_call_result_6981)
        
        # Assigning a Call to a Attribute (line 300):
        
        # Call to float_to_str(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'epsneg' (line 300)
        epsneg_6984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 40), 'epsneg', False)
        # Processing the call keyword arguments (line 300)
        kwargs_6985 = {}
        # Getting the type of 'float_to_str' (line 300)
        float_to_str_6983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'float_to_str', False)
        # Calling float_to_str(args, kwargs) (line 300)
        float_to_str_call_result_6986 = invoke(stypy.reporting.localization.Localization(__file__, 300, 27), float_to_str_6983, *[epsneg_6984], **kwargs_6985)
        
        # Getting the type of 'self' (line 300)
        self_6987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'self')
        # Setting the type of the member '_str_epsneg' of a type (line 300)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), self_6987, '_str_epsneg', float_to_str_call_result_6986)
        
        # Assigning a Name to a Attribute (line 301):
        # Getting the type of 'machep' (line 301)
        machep_6988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 22), 'machep')
        # Getting the type of 'self' (line 301)
        self_6989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'self')
        # Setting the type of the member 'machep' of a type (line 301)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), self_6989, 'machep', machep_6988)
        
        # Assigning a Call to a Attribute (line 302):
        
        # Call to float_to_float(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'eps' (line 302)
        eps_6991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 34), 'eps', False)
        # Processing the call keyword arguments (line 302)
        kwargs_6992 = {}
        # Getting the type of 'float_to_float' (line 302)
        float_to_float_6990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 19), 'float_to_float', False)
        # Calling float_to_float(args, kwargs) (line 302)
        float_to_float_call_result_6993 = invoke(stypy.reporting.localization.Localization(__file__, 302, 19), float_to_float_6990, *[eps_6991], **kwargs_6992)
        
        # Getting the type of 'self' (line 302)
        self_6994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'self')
        # Setting the type of the member 'eps' of a type (line 302)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), self_6994, 'eps', float_to_float_call_result_6993)
        
        # Assigning a Call to a Attribute (line 303):
        
        # Call to float_to_str(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'eps' (line 303)
        eps_6996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 37), 'eps', False)
        # Processing the call keyword arguments (line 303)
        kwargs_6997 = {}
        # Getting the type of 'float_to_str' (line 303)
        float_to_str_6995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'float_to_str', False)
        # Calling float_to_str(args, kwargs) (line 303)
        float_to_str_call_result_6998 = invoke(stypy.reporting.localization.Localization(__file__, 303, 24), float_to_str_6995, *[eps_6996], **kwargs_6997)
        
        # Getting the type of 'self' (line 303)
        self_6999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'self')
        # Setting the type of the member '_str_eps' of a type (line 303)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), self_6999, '_str_eps', float_to_str_call_result_6998)
        
        # Assigning a Name to a Attribute (line 304):
        # Getting the type of 'ngrd' (line 304)
        ngrd_7000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'ngrd')
        # Getting the type of 'self' (line 304)
        self_7001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'self')
        # Setting the type of the member 'ngrd' of a type (line 304)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), self_7001, 'ngrd', ngrd_7000)
        
        # Assigning a Name to a Attribute (line 305):
        # Getting the type of 'iexp' (line 305)
        iexp_7002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'iexp')
        # Getting the type of 'self' (line 305)
        self_7003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'self')
        # Setting the type of the member 'iexp' of a type (line 305)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), self_7003, 'iexp', iexp_7002)
        
        # Assigning a Name to a Attribute (line 306):
        # Getting the type of 'minexp' (line 306)
        minexp_7004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'minexp')
        # Getting the type of 'self' (line 306)
        self_7005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self')
        # Setting the type of the member 'minexp' of a type (line 306)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_7005, 'minexp', minexp_7004)
        
        # Assigning a Call to a Attribute (line 307):
        
        # Call to float_to_float(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'xmin' (line 307)
        xmin_7007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 35), 'xmin', False)
        # Processing the call keyword arguments (line 307)
        kwargs_7008 = {}
        # Getting the type of 'float_to_float' (line 307)
        float_to_float_7006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'float_to_float', False)
        # Calling float_to_float(args, kwargs) (line 307)
        float_to_float_call_result_7009 = invoke(stypy.reporting.localization.Localization(__file__, 307, 20), float_to_float_7006, *[xmin_7007], **kwargs_7008)
        
        # Getting the type of 'self' (line 307)
        self_7010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self')
        # Setting the type of the member 'xmin' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_7010, 'xmin', float_to_float_call_result_7009)
        
        # Assigning a Call to a Attribute (line 308):
        
        # Call to float_to_str(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'xmin' (line 308)
        xmin_7012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 38), 'xmin', False)
        # Processing the call keyword arguments (line 308)
        kwargs_7013 = {}
        # Getting the type of 'float_to_str' (line 308)
        float_to_str_7011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'float_to_str', False)
        # Calling float_to_str(args, kwargs) (line 308)
        float_to_str_call_result_7014 = invoke(stypy.reporting.localization.Localization(__file__, 308, 25), float_to_str_7011, *[xmin_7012], **kwargs_7013)
        
        # Getting the type of 'self' (line 308)
        self_7015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self')
        # Setting the type of the member '_str_xmin' of a type (line 308)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_7015, '_str_xmin', float_to_str_call_result_7014)
        
        # Assigning a Name to a Attribute (line 309):
        # Getting the type of 'maxexp' (line 309)
        maxexp_7016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 22), 'maxexp')
        # Getting the type of 'self' (line 309)
        self_7017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self')
        # Setting the type of the member 'maxexp' of a type (line 309)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_7017, 'maxexp', maxexp_7016)
        
        # Assigning a Call to a Attribute (line 310):
        
        # Call to float_to_float(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'xmax' (line 310)
        xmax_7019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 35), 'xmax', False)
        # Processing the call keyword arguments (line 310)
        kwargs_7020 = {}
        # Getting the type of 'float_to_float' (line 310)
        float_to_float_7018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'float_to_float', False)
        # Calling float_to_float(args, kwargs) (line 310)
        float_to_float_call_result_7021 = invoke(stypy.reporting.localization.Localization(__file__, 310, 20), float_to_float_7018, *[xmax_7019], **kwargs_7020)
        
        # Getting the type of 'self' (line 310)
        self_7022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self')
        # Setting the type of the member 'xmax' of a type (line 310)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_7022, 'xmax', float_to_float_call_result_7021)
        
        # Assigning a Call to a Attribute (line 311):
        
        # Call to float_to_str(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'xmax' (line 311)
        xmax_7024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 38), 'xmax', False)
        # Processing the call keyword arguments (line 311)
        kwargs_7025 = {}
        # Getting the type of 'float_to_str' (line 311)
        float_to_str_7023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 25), 'float_to_str', False)
        # Calling float_to_str(args, kwargs) (line 311)
        float_to_str_call_result_7026 = invoke(stypy.reporting.localization.Localization(__file__, 311, 25), float_to_str_7023, *[xmax_7024], **kwargs_7025)
        
        # Getting the type of 'self' (line 311)
        self_7027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'self')
        # Setting the type of the member '_str_xmax' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), self_7027, '_str_xmax', float_to_str_call_result_7026)
        
        # Assigning a Name to a Attribute (line 312):
        # Getting the type of 'irnd' (line 312)
        irnd_7028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'irnd')
        # Getting the type of 'self' (line 312)
        self_7029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self')
        # Setting the type of the member 'irnd' of a type (line 312)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_7029, 'irnd', irnd_7028)
        
        # Assigning a Name to a Attribute (line 314):
        # Getting the type of 'title' (line 314)
        title_7030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 21), 'title')
        # Getting the type of 'self' (line 314)
        self_7031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self')
        # Setting the type of the member 'title' of a type (line 314)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_7031, 'title', title_7030)
        
        # Assigning a Attribute to a Attribute (line 316):
        # Getting the type of 'self' (line 316)
        self_7032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'self')
        # Obtaining the member 'eps' of a type (line 316)
        eps_7033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 23), self_7032, 'eps')
        # Getting the type of 'self' (line 316)
        self_7034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self')
        # Setting the type of the member 'epsilon' of a type (line 316)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_7034, 'epsilon', eps_7033)
        
        # Assigning a Attribute to a Attribute (line 317):
        # Getting the type of 'self' (line 317)
        self_7035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'self')
        # Obtaining the member 'xmin' of a type (line 317)
        xmin_7036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 20), self_7035, 'xmin')
        # Getting the type of 'self' (line 317)
        self_7037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self')
        # Setting the type of the member 'tiny' of a type (line 317)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_7037, 'tiny', xmin_7036)
        
        # Assigning a Attribute to a Attribute (line 318):
        # Getting the type of 'self' (line 318)
        self_7038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'self')
        # Obtaining the member 'xmax' of a type (line 318)
        xmax_7039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 20), self_7038, 'xmax')
        # Getting the type of 'self' (line 318)
        self_7040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'self')
        # Setting the type of the member 'huge' of a type (line 318)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), self_7040, 'huge', xmax_7039)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 320, 8))
        
        # 'import math' statement (line 320)
        import math

        import_module(stypy.reporting.localization.Localization(__file__, 320, 8), 'math', math, module_type_store)
        
        
        # Assigning a Call to a Attribute (line 321):
        
        # Call to int(...): (line 321)
        # Processing the call arguments (line 321)
        
        
        # Call to log10(...): (line 321)
        # Processing the call arguments (line 321)
        
        # Call to float_to_float(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'self' (line 321)
        self_7045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 56), 'self', False)
        # Obtaining the member 'eps' of a type (line 321)
        eps_7046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 56), self_7045, 'eps')
        # Processing the call keyword arguments (line 321)
        kwargs_7047 = {}
        # Getting the type of 'float_to_float' (line 321)
        float_to_float_7044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 41), 'float_to_float', False)
        # Calling float_to_float(args, kwargs) (line 321)
        float_to_float_call_result_7048 = invoke(stypy.reporting.localization.Localization(__file__, 321, 41), float_to_float_7044, *[eps_7046], **kwargs_7047)
        
        # Processing the call keyword arguments (line 321)
        kwargs_7049 = {}
        # Getting the type of 'math' (line 321)
        math_7042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'math', False)
        # Obtaining the member 'log10' of a type (line 321)
        log10_7043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 30), math_7042, 'log10')
        # Calling log10(args, kwargs) (line 321)
        log10_call_result_7050 = invoke(stypy.reporting.localization.Localization(__file__, 321, 30), log10_7043, *[float_to_float_call_result_7048], **kwargs_7049)
        
        # Applying the 'usub' unary operator (line 321)
        result___neg___7051 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 29), 'usub', log10_call_result_7050)
        
        # Processing the call keyword arguments (line 321)
        kwargs_7052 = {}
        # Getting the type of 'int' (line 321)
        int_7041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 25), 'int', False)
        # Calling int(args, kwargs) (line 321)
        int_call_result_7053 = invoke(stypy.reporting.localization.Localization(__file__, 321, 25), int_7041, *[result___neg___7051], **kwargs_7052)
        
        # Getting the type of 'self' (line 321)
        self_7054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self')
        # Setting the type of the member 'precision' of a type (line 321)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_7054, 'precision', int_call_result_7053)
        
        # Assigning a BinOp to a Name (line 322):
        # Getting the type of 'two' (line 322)
        two_7055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 14), 'two')
        # Getting the type of 'two' (line 322)
        two_7056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'two')
        # Applying the binary operator '+' (line 322)
        result_add_7057 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 14), '+', two_7055, two_7056)
        
        # Getting the type of 'two' (line 322)
        two_7058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 26), 'two')
        # Applying the binary operator '+' (line 322)
        result_add_7059 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 24), '+', result_add_7057, two_7058)
        
        # Getting the type of 'two' (line 322)
        two_7060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 32), 'two')
        # Applying the binary operator '+' (line 322)
        result_add_7061 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 30), '+', result_add_7059, two_7060)
        
        # Getting the type of 'two' (line 322)
        two_7062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 38), 'two')
        # Applying the binary operator '+' (line 322)
        result_add_7063 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 36), '+', result_add_7061, two_7062)
        
        # Assigning a type to the variable 'ten' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'ten', result_add_7063)
        
        # Assigning a BinOp to a Name (line 323):
        # Getting the type of 'ten' (line 323)
        ten_7064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 21), 'ten')
        
        # Getting the type of 'self' (line 323)
        self_7065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 30), 'self')
        # Obtaining the member 'precision' of a type (line 323)
        precision_7066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 30), self_7065, 'precision')
        # Applying the 'usub' unary operator (line 323)
        result___neg___7067 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 29), 'usub', precision_7066)
        
        # Applying the binary operator '**' (line 323)
        result_pow_7068 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 21), '**', ten_7064, result___neg___7067)
        
        # Assigning a type to the variable 'resolution' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'resolution', result_pow_7068)
        
        # Assigning a Call to a Attribute (line 324):
        
        # Call to float_to_float(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'resolution' (line 324)
        resolution_7070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 41), 'resolution', False)
        # Processing the call keyword arguments (line 324)
        kwargs_7071 = {}
        # Getting the type of 'float_to_float' (line 324)
        float_to_float_7069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 26), 'float_to_float', False)
        # Calling float_to_float(args, kwargs) (line 324)
        float_to_float_call_result_7072 = invoke(stypy.reporting.localization.Localization(__file__, 324, 26), float_to_float_7069, *[resolution_7070], **kwargs_7071)
        
        # Getting the type of 'self' (line 324)
        self_7073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'self')
        # Setting the type of the member 'resolution' of a type (line 324)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), self_7073, 'resolution', float_to_float_call_result_7072)
        
        # Assigning a Call to a Attribute (line 325):
        
        # Call to float_to_str(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'resolution' (line 325)
        resolution_7075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 44), 'resolution', False)
        # Processing the call keyword arguments (line 325)
        kwargs_7076 = {}
        # Getting the type of 'float_to_str' (line 325)
        float_to_str_7074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 31), 'float_to_str', False)
        # Calling float_to_str(args, kwargs) (line 325)
        float_to_str_call_result_7077 = invoke(stypy.reporting.localization.Localization(__file__, 325, 31), float_to_str_7074, *[resolution_7075], **kwargs_7076)
        
        # Getting the type of 'self' (line 325)
        self_7078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'self')
        # Setting the type of the member '_str_resolution' of a type (line 325)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), self_7078, '_str_resolution', float_to_str_call_result_7077)
        
        # ################# End of '_do_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_do_init' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_7079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7079)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_do_init'
        return stypy_return_type_7079


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MachAr.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        MachAr.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MachAr.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MachAr.stypy__str__.__dict__.__setitem__('stypy_function_name', 'MachAr.__str__')
        MachAr.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        MachAr.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MachAr.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MachAr.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MachAr.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MachAr.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MachAr.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MachAr.__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Assigning a Str to a Name (line 328):
        str_7080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 11), 'str', 'Machine parameters for %(title)s\n---------------------------------------------------------------------\nibeta=%(ibeta)s it=%(it)s iexp=%(iexp)s ngrd=%(ngrd)s irnd=%(irnd)s\nmachep=%(machep)s     eps=%(_str_eps)s (beta**machep == epsilon)\nnegep =%(negep)s  epsneg=%(_str_epsneg)s (beta**epsneg)\nminexp=%(minexp)s   xmin=%(_str_xmin)s (beta**minexp == tiny)\nmaxexp=%(maxexp)s    xmax=%(_str_xmax)s ((1-epsneg)*beta**maxexp == huge)\n---------------------------------------------------------------------\n')
        # Assigning a type to the variable 'fmt' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'fmt', str_7080)
        # Getting the type of 'fmt' (line 338)
        fmt_7081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'fmt')
        # Getting the type of 'self' (line 338)
        self_7082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 21), 'self')
        # Obtaining the member '__dict__' of a type (line 338)
        dict___7083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 21), self_7082, '__dict__')
        # Applying the binary operator '%' (line 338)
        result_mod_7084 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 15), '%', fmt_7081, dict___7083)
        
        # Assigning a type to the variable 'stypy_return_type' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'stypy_return_type', result_mod_7084)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 327)
        stypy_return_type_7085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_7085


# Assigning a type to the variable 'MachAr' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'MachAr', MachAr)

if (__name__ == '__main__'):
    
    # Call to print(...): (line 342)
    # Processing the call arguments (line 342)
    
    # Call to MachAr(...): (line 342)
    # Processing the call keyword arguments (line 342)
    kwargs_7088 = {}
    # Getting the type of 'MachAr' (line 342)
    MachAr_7087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 10), 'MachAr', False)
    # Calling MachAr(args, kwargs) (line 342)
    MachAr_call_result_7089 = invoke(stypy.reporting.localization.Localization(__file__, 342, 10), MachAr_7087, *[], **kwargs_7088)
    
    # Processing the call keyword arguments (line 342)
    kwargs_7090 = {}
    # Getting the type of 'print' (line 342)
    print_7086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'print', False)
    # Calling print(args, kwargs) (line 342)
    print_call_result_7091 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), print_7086, *[MachAr_call_result_7089], **kwargs_7090)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
